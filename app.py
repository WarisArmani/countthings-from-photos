
import io
import hashlib

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


# ============================================================
# APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Classroom Head Counter",
    page_icon="👥",
    layout="wide",
)

MODEL_REPO = "wesjos/YOLO-head-detection"
MODEL_FILE = "v8s-head.pt"

MAX_UPLOAD_MB = 20
MAX_IMAGE_PIXELS = 25_000_000

# Smaller tiles enlarge heads relative to the tile during inference.
TILE_SIZE = 512

# More overlap gives heads near tile edges additional opportunities
# to be detected.
TILE_OVERLAP = 0.40

# Run detection at multiple inference sizes.
INFERENCE_SIZES = [1280, 960]

# Detection settings.
MODEL_IOU = 0.60
MERGE_IOU = 0.60
MAX_DETECTIONS_PER_TILE = 500


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading head detection model...")
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
    )

    # Do not assign model.names; this property is read-only.
    return YOLO(model_path)


# ============================================================
# IMAGE VALIDATION
# ============================================================

def open_uploaded_image(file_bytes):
    with Image.open(io.BytesIO(file_bytes)) as source:
        image = ImageOps.exif_transpose(source)
        image.load()
        image = image.convert("RGB")

    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError(
            "The image is too large. Please resize it and upload it again."
        )

    return image


# ============================================================
# IMAGE TILING
# ============================================================

def get_positions(length, tile_size, overlap):
    if length <= tile_size:
        return [0]

    stride = max(1, int(tile_size * (1 - overlap)))

    positions = list(
        range(0, length - tile_size + 1, stride)
    )

    final_position = length - tile_size

    if positions[-1] != final_position:
        positions.append(final_position)

    return positions


def make_tiles(image):
    width, height = image.size

    x_positions = get_positions(
        width,
        TILE_SIZE,
        TILE_OVERLAP,
    )

    y_positions = get_positions(
        height,
        TILE_SIZE,
        TILE_OVERLAP,
    )

    tiles = []

    for y in y_positions:
        for x in x_positions:
            right = min(x + TILE_SIZE, width)
            bottom = min(y + TILE_SIZE, height)

            tile = image.crop((x, y, right, bottom))

            tiles.append({
                "image": tile,
                "x": x,
                "y": y,
            })

    return tiles


# ============================================================
# DUPLICATE REMOVAL
# ============================================================

def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0.0,
        intersection_x2 - intersection_x1,
    )

    intersection_height = max(
        0.0,
        intersection_y2 - intersection_y1,
    )

    intersection = intersection_width * intersection_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def remove_duplicates(detections, iou_threshold=MERGE_IOU):
    """
    Merge duplicate detections from overlapping tiles and
    different inference sizes.

    Higher-confidence detections are retained first.
    """

    if not detections:
        return []

    ordered = sorted(
        detections,
        key=lambda item: item["confidence"],
        reverse=True,
    )

    kept = []

    for candidate in ordered:
        duplicate = False

        for existing in kept:
            overlap = calculate_iou(
                candidate["box"],
                existing["box"],
            )

            if overlap >= iou_threshold:
                duplicate = True
                break

        if not duplicate:
            kept.append(candidate)

    return kept


# ============================================================
# FONT AND ANNOTATION
# ============================================================

def load_font(size=16):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def annotate_image(image, detections):
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)

    font = load_font(16)

    for number, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection["box"]
        confidence = detection["confidence"]

        box = (
            int(x1),
            int(y1),
            int(x2),
            int(y2),
        )

        draw.rectangle(
            box,
            outline=(0, 255, 80),
            width=3,
        )

        label = f"{number} ({confidence:.2f})"

        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]

        label_x = max(0, int(x1))
        label_y = max(0, int(y1) - text_height - 8)

        draw.rectangle(
            (
                label_x,
                label_y,
                label_x + text_width + 8,
                label_y + text_height + 6,
            ),
            fill=(0, 140, 40),
        )

        draw.text(
            (label_x + 4, label_y + 3),
            label,
            fill="white",
            font=font,
        )

    return annotated


# ============================================================
# HEAD DETECTION
# ============================================================

def detect_heads(image, model, confidence, use_multiscale):
    image_width, image_height = image.size

    tiles = make_tiles(image)

    scales = INFERENCE_SIZES if use_multiscale else [INFERENCE_SIZES[0]]

    total_steps = len(tiles) * len(scales)
    completed_steps = 0

    detections = []

    progress = st.progress(
        0,
        text="Preparing classroom image...",
    )

    try:
        for tile_number, tile_info in enumerate(tiles, start=1):
            tile = tile_info["image"]
            offset_x = tile_info["x"]
            offset_y = tile_info["y"]

            # Convert to RGB before passing the image to YOLO.
            tile_array = np.asarray(tile.convert("RGB"))

            for inference_size in scales:
                results = model.predict(
                    source=tile_array,
                    imgsz=inference_size,
                    conf=float(confidence),
                    iou=MODEL_IOU,
                    max_det=MAX_DETECTIONS_PER_TILE,
                    classes=[0],
                    device="cpu",
                    verbose=False,
                )

                result = results[0]

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()

                    for box, score in zip(boxes, scores):
                        x1, y1, x2, y2 = map(float, box)

                        # Convert tile coordinates to original coordinates.
                        x1 += offset_x
                        x2 += offset_x
                        y1 += offset_y
                        y2 += offset_y

                        # Keep boxes inside the original image.
                        x1 = max(0.0, min(x1, image_width - 1))
                        y1 = max(0.0, min(y1, image_height - 1))
                        x2 = max(0.0, min(x2, image_width - 1))
                        y2 = max(0.0, min(y2, image_height - 1))

                        if x2 <= x1 or y2 <= y1:
                            continue

                        detections.append({
                            "box": [x1, y1, x2, y2],
                            "confidence": float(score),
                        })

                completed_steps += 1

                progress.progress(
                    min(completed_steps / total_steps, 1.0),
                    text=(
                        f"Analyzing tile {tile_number}/{len(tiles)} "
                        f"at {inference_size}px"
                    ),
                )

    finally:
        progress.empty()

    raw_detection_count = len(detections)

    final_detections = remove_duplicates(
        detections,
        iou_threshold=MERGE_IOU,
    )

    final_detections.sort(
        key=lambda item: (
            (item["box"][1] + item["box"][3]) / 2,
            (item["box"][0] + item["box"][2]) / 2,
        ),
    )

    annotated_image = annotate_image(
        image,
        final_detections,
    )

    return {
        "count": len(final_detections),
        "annotated_image": annotated_image,
        "tile_count": len(tiles),
        "raw_detections": raw_detection_count,
        "detections": final_detections,
    }


# ============================================================
# STREAMLIT INTERFACE
# ============================================================

st.title("Classroom Head Counter")

st.write(
    "Count visible heads in classroom photographs, including "
    "heads covered by scarves or hijabs when the model can recognize them."
)

with st.sidebar:
    st.header("Detection Settings")

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.80,
        value=0.05,
        step=0.05,
        help=(
            "A lower threshold accepts less-confident detections. "
            "It can find more heads, but may also add false detections."
        ),
    )

    use_multiscale = st.checkbox(
        "Use multi-scale detection",
        value=True,
        help=(
            "Runs each tile at two inference sizes to help detect "
            "heads of different sizes."
        ),
    )

    st.caption(f"Tile size: {TILE_SIZE}px")
    st.caption(f"Tile overlap: {TILE_OVERLAP:.0%}")

uploaded_file = st.file_uploader(
    "Upload a classroom photo",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()

    if len(file_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(
            f"Please upload an image smaller than {MAX_UPLOAD_MB} MB."
        )
        st.stop()

    file_hash = hashlib.sha256(file_bytes).hexdigest()

    # Clear previous results when a different image is uploaded.
    previous_hash = st.session_state.get("image_hash")

    if previous_hash != file_hash:
        st.session_state["image_hash"] = file_hash
        st.session_state.pop("detection_result", None)

    try:
        image = open_uploaded_image(file_bytes)
    except Exception as error:
        st.error(f"Could not read the image: {error}")
        st.stop()

    st.subheader("Original Classroom Photo")

    st.image(
        image,
        caption=f"{image.width} × {image.height} pixels",
        use_container_width=True,
    )

    run_detection = st.button(
        "Count Heads",
        type="primary",
        use_container_width=True,
    )

    if run_detection:
        try:
            model = load_model()

            with st.spinner(
                "Detecting heads across the classroom image..."
            ):
                result = detect_heads(
                    image=image,
                    model=model,
                    confidence=confidence,
                    use_multiscale=use_multiscale,
                )

            result["image_hash"] = file_hash
            result["confidence_setting"] = confidence
            result["multiscale_setting"] = use_multiscale

            st.session_state["detection_result"] = result

        except Exception as error:
            st.error(f"Head detection failed: {error}")

    result = st.session_state.get("detection_result")

    if result and result["image_hash"] == file_hash:
        settings_match = (
            result["confidence_setting"] == confidence
            and result["multiscale_setting"] == use_multiscale
        )

        if not settings_match:
            st.warning(
                "Your detection settings have changed. Click "
                "'Count Heads' again to run detection with the new settings."
            )
        else:
            st.divider()
            st.subheader("Detection Results")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Estimated Head Count",
                result["count"],
            )

            col2.metric(
                "Image Tiles",
                result["tile_count"],
            )

            col3.metric(
                "Detections Before Duplicate Removal",
                result["raw_detections"],
            )

            st.image(
                result["annotated_image"],
                caption="Green boxes mark detected heads.",
                use_container_width=True,
            )

            image_output = io.BytesIO()

            result["annotated_image"].save(
                image_output,
                format="JPEG",
                quality=95,
            )

            st.download_button(
                "Download Annotated Photo",
                data=image_output.getvalue(),
                file_name="classroom_heads_detected.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )

            st.info(
                "Review the green boxes. A missed scarf-covered head "
                "means the model did not detect it; a box around a "
                "non-head object is a false positive."
            )

st.divider()

st.caption(
    "Accuracy depends on the model's training data, image resolution, "
    "lighting, crowding, and how much each head is obscured. "
    "This app estimates visible heads; it does not guarantee an exact count."
)
