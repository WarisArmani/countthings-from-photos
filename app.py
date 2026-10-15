
import io
import math
import streamlit as st
import numpy as np

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

# Smaller tiles can help detect small or crowded heads.
TILE_SIZE = 512
TILE_OVERLAP = 0.35

# Higher inference size may help with small heads, but is slower.
INFERENCE_SIZE = 1280

# Model-level NMS and cross-tile duplicate removal.
MODEL_IOU = 0.60
MERGE_IOU = 0.55
MAX_DETECTIONS_PER_TILE = 500


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner="Loading head detection model...")
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
    )

    # Do not assign model.names; it is a read-only property.
    return YOLO(model_path)


# ============================================================
# IMAGE TILING
# ============================================================

def get_tile_positions(image_length, tile_size, overlap):
    """Return tile start positions and ensure the far edge is covered."""

    if image_length <= tile_size:
        return [0]

    stride = max(1, int(tile_size * (1 - overlap)))
    positions = list(range(0, image_length - tile_size + 1, stride))

    final_position = image_length - tile_size

    if positions[-1] != final_position:
        positions.append(final_position)

    return positions


def make_tiles(image, tile_size=TILE_SIZE, overlap=TILE_OVERLAP):
    """Split the image into overlapping tiles."""

    width, height = image.size

    x_positions = get_tile_positions(width, tile_size, overlap)
    y_positions = get_tile_positions(height, tile_size, overlap)

    tiles = []

    for y in y_positions:
        for x in x_positions:
            x2 = min(x + tile_size, width)
            y2 = min(y + tile_size, height)

            tile = image.crop((x, y, x2, y2))
            tiles.append((tile, x, y))

    return tiles


# ============================================================
# BOX UTILITIES
# ============================================================

def calculate_iou(box_a, box_b):
    """Calculate intersection over union for two xyxy boxes."""

    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)
    intersection = intersection_width * intersection_height

    area_a = max(0.0, box_a[2] - box_a[0]) * max(
        0.0, box_a[3] - box_a[1]
    )
    area_b = max(0.0, box_b[2] - box_b[0]) * max(
        0.0, box_b[3] - box_b[1]
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def remove_duplicates(detections, iou_threshold=MERGE_IOU):
    """
    Remove repeated detections caused by overlapping tiles.

    Detections with higher confidence are kept first.
    """

    if not detections:
        return []

    detections = sorted(
        detections,
        key=lambda detection: detection["confidence"],
        reverse=True,
    )

    kept = []

    for detection in detections:
        is_duplicate = False

        for existing in kept:
            overlap = calculate_iou(
                detection["box"],
                existing["box"],
            )

            if overlap >= iou_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(detection)

    return kept


# ============================================================
# HEAD DETECTION
# ============================================================

def detect_heads(
    image,
    model,
    confidence=0.05,
    use_multiscale=True,
):
    """
    Detect heads on overlapping tiles and optionally run a second
    inference pass at a different image size.
    """

    image_width, image_height = image.size
    tiles = make_tiles(image)

    all_detections = []
    inference_sizes = (
        [INFERENCE_SIZE, 960] if use_multiscale else [INFERENCE_SIZE]
    )

    progress = st.progress(0, text="Preparing image...")

    total_steps = max(1, len(tiles) * len(inference_sizes))
    completed_steps = 0

    for tile_index, (tile, offset_x, offset_y) in enumerate(tiles):
        tile_array = np.asarray(tile.convert("RGB"))

        for inference_size in inference_sizes:
            try:
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
                        x1, y1, x2, y2 = box.tolist()

                        # Convert tile coordinates to original-image coordinates.
                        x1 += offset_x
                        x2 += offset_x
                        y1 += offset_y
                        y2 += offset_y

                        # Clip coordinates to image boundaries.
                        x1 = max(0.0, min(x1, image_width - 1))
                        y1 = max(0.0, min(y1, image_height - 1))
                        x2 = max(0.0, min(x2, image_width - 1))
                        y2 = max(0.0, min(y2, image_height - 1))

                        if x2 <= x1 or y2 <= y1:
                            continue

                        all_detections.append({
                            "box": [x1, y1, x2, y2],
                            "confidence": float(score),
                        })

            except Exception as error:
                progress.empty()
                raise RuntimeError(
                    f"Detection failed on tile {tile_index + 1}: {error}"
                ) from error

            completed_steps += 1
            progress.progress(
                min(completed_steps / total_steps, 1.0),
                text=(
                    f"Analyzing tile {tile_index + 1}/{len(tiles)} "
                    f"({inference_size}px inference)"
                ),
            )

    progress.empty()

    # Merge repeated detections from overlapping tiles and scale passes.
    final_detections = remove_duplicates(
        all_detections,
        iou_threshold=MERGE_IOU,
    )

    # Draw results on a copy of the original image.
    annotated_image = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated_image)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for index, detection in enumerate(final_detections, start=1):
        x1, y1, x2, y2 = detection["box"]
        score = detection["confidence"]

        draw.rectangle(
            (int(x1), int(y1), int(x2), int(y2)),
            outline=(0, 255, 80),
            width=3,
        )

        label = f"{index}: {score:.2f}"

        text_box = draw.textbbox((0, 0), label, font=font)
        label_width = text_box[2] - text_box[0]
        label_height = text_box[3] - text_box[1]

        label_x = max(0, int(x1))
        label_y = max(0, int(y1) - label_height - 6)

        draw.rectangle(
            (
                label_x,
                label_y,
                label_x + label_width + 8,
                label_y + label_height + 6,
            ),
            fill=(0, 150, 50),
        )

        draw.text(
            (label_x + 4, label_y + 3),
            label,
            fill=(255, 255, 255),
            font=font,
        )

    return (
        len(final_detections),
        annotated_image,
        len(tiles),
        len(all_detections),
    )


# ============================================================
# USER INTERFACE
# ============================================================

st.title("👥 Classroom Head Counter")

st.write(
    "Upload a classroom photo to detect and count visible heads. "
    "The app analyzes overlapping image tiles to help find smaller heads."
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
            "Lower values may find more heads but can also increase "
            "false detections."
        ),
    )

    use_multiscale = st.checkbox(
        "Use multi-scale detection",
        value=True,
        help=(
            "Runs detection at two inference sizes. This may improve "
            "small-head detection but takes longer."
        ),
    )

    st.caption(f"Tile size: {TILE_SIZE} × {TILE_SIZE}")
    st.caption(f"Tile overlap: {TILE_OVERLAP:.0%}")
    st.caption(f"Inference size: {INFERENCE_SIZE}")

uploaded_file = st.file_uploader(
    "Upload a classroom photo",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()

    if len(file_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"Please upload an image smaller than {MAX_UPLOAD_MB} MB.")
        st.stop()

    try:
        with Image.open(io.BytesIO(file_bytes)) as opened_image:
            image = ImageOps.exif_transpose(opened_image)
            image.load()
            image = image.convert("RGB")

    except Exception:
        st.error("The uploaded file could not be opened as an image.")
        st.stop()

    if image.width * image.height > MAX_IMAGE_PIXELS:
        st.error(
            "This image has too many pixels. Please upload a smaller "
            "or resized image."
        )
        st.stop()

    st.subheader("Original Photo")

    st.image(
        image,
        caption=f"Image dimensions: {image.width} × {image.height}",
        use_container_width=True,
    )

    if st.button("Count Heads", type="primary", use_container_width=True):
        try:
            model = load_model()

            count, annotated_image, tile_count, raw_count = detect_heads(
                image=image,
                model=model,
                confidence=confidence,
                use_multiscale=use_multiscale,
            )

            st.session_state["head_count_result"] = {
                "count": count,
                "annotated_image": annotated_image,
                "tile_count": tile_count,
                "raw_count": raw_count,
                "filename": uploaded_file.name,
            }

        except Exception as error:
            st.error(f"Head detection failed: {error}")

    result = st.session_state.get("head_count_result")

    if result is not None and result["filename"] == uploaded_file.name:
        st.divider()
        st.subheader("Detection Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Estimated Heads", result["count"])
        col2.metric("Image Tiles", result["tile_count"])
        col3.metric("Detections Before Deduplication", result["raw_count"])

        st.image(
            result["annotated_image"],
            caption="Green boxes indicate detected heads.",
            use_container_width=True,
        )

        output = io.BytesIO()
        result["annotated_image"].save(
            output,
            format="JPEG",
            quality=95,
        )

        st.download_button(
            label="Download Annotated Photo",
            data=output.getvalue(),
            file_name="classroom_heads_detected.jpg",
            mime="image/jpeg",
            use_container_width=True,
        )

        st.info(
            "Please check the green boxes against the actual heads. "
            "A box around an object that is not a head is a false positive; "
            "a visible head without a box is a missed detection."
        )

st.divider()

st.caption(
    "Note: The count is an estimate from the detection model. "
    "Accuracy depends on photo quality, head size, occlusion, and crowding. "
    "Review the annotated photo before using the count in a report."
)
