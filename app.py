
import io
import hashlib
import time

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageDraw, UnidentifiedImageError
from huggingface_hub import hf_hub_download


# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------

APP_TITLE = "Classroom Head Counter"

MODEL_REPO = "AmineSam/irail-crowd-counting-yolov8n"
MODEL_FILE = "best.pt"

MAX_UPLOAD_MB = 20
MAX_IMAGE_PIXELS = 25_000_000

TILE_SIZE = 640
TILE_OVERLAP = 0.25
TILE_IMAGE_SIZE = 960

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="👥",
    layout="wide",
)

st.title("Classroom Head Counter")
st.write(
    "Upload a classroom or group photo to estimate the number "
    "of visible heads. Overlapping image tiles help the model "
    "inspect smaller and distant heads."
)

st.info(
    "AI counts are estimates. Check the marked image for missed "
    "heads and duplicate detections before using the result."
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model():
    from ultralytics import YOLO

    weights_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
    )

    return YOLO(weights_path)


# --------------------------------------------------
# IMAGE HELPERS
# --------------------------------------------------

def read_uploaded_image(raw_bytes):
    with Image.open(io.BytesIO(raw_bytes)) as source:
        width, height = source.size

        if width < 1 or height < 1:
            raise ValueError("The image has invalid dimensions.")

        if width * height > MAX_IMAGE_PIXELS:
            raise ValueError(
                "The image exceeds 25 million pixels. "
                "Please upload a smaller image."
            )

        return ImageOps.exif_transpose(source).convert("RGB")


def image_to_jpeg_bytes(image):
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=92, optimize=True)
    return output.getvalue()


# --------------------------------------------------
# CREATE OVERLAPPING TILES
# --------------------------------------------------

def make_tiles(image, tile_size=TILE_SIZE, overlap=TILE_OVERLAP):
    width, height = image.size
    step = max(1, int(tile_size * (1 - overlap)))

    def get_positions(length):
        if length <= tile_size:
            return [0]

        positions = list(range(0, length - tile_size + 1, step))
        last_position = length - tile_size

        if positions[-1] != last_position:
            positions.append(last_position)

        return positions

    tiles = []

    for y in get_positions(height):
        for x in get_positions(width):
            right = min(x + tile_size, width)
            bottom = min(y + tile_size, height)

            tile = image.crop((x, y, right, bottom))
            tiles.append((x, y, tile))

    return tiles


# --------------------------------------------------
# NON-MAXIMUM SUPPRESSION
# Remove overlapping duplicate boxes without relying
# on torchvision.ops.nms.
# --------------------------------------------------

def box_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection_width = np.maximum(0, x2 - x1)
    intersection_height = np.maximum(0, y2 - y1)
    intersection = intersection_width * intersection_height

    area_box = max(0, box[2] - box[0]) * max(
        0, box[3] - box[1]
    )

    area_boxes = np.maximum(
        0, boxes[:, 2] - boxes[:, 0]
    ) * np.maximum(
        0, boxes[:, 3] - boxes[:, 1]
    )

    union = area_box + area_boxes - intersection

    return intersection / np.maximum(union, 1e-9)


def remove_duplicate_boxes(boxes, scores, iou_threshold=0.45):
    if len(boxes) == 0:
        return []

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    # Process high-confidence detections first.
    order = np.argsort(scores)[::-1]
    kept = []

    while len(order) > 0:
        current = order[0]
        kept.append(int(current))

        if len(order) == 1:
            break

        remaining = order[1:]
        overlaps = box_iou(boxes[current], boxes[remaining])

        order = remaining[overlaps < iou_threshold]

    return kept


# --------------------------------------------------
# RUN TILED HEAD DETECTION
# --------------------------------------------------

def detect_heads_tiled(model, image, confidence):
    boxes_all = []
    scores_all = []

    tiles = make_tiles(image)
    progress = st.progress(0, text="Preparing image tiles...")

    try:
        for tile_number, (x_offset, y_offset, tile) in enumerate(tiles):
            prediction = model.predict(
                source=np.asarray(tile),
                imgsz=TILE_IMAGE_SIZE,
                conf=float(confidence),
                iou=0.60,
                max_det=300,
                device="cpu",
                verbose=False,
            )[0]

            if prediction.boxes is not None and len(prediction.boxes):
                tile_boxes = prediction.boxes.xyxy.cpu().numpy()
                tile_scores = prediction.boxes.conf.cpu().numpy()

                tile_width, tile_height = tile.size

                for box, score in zip(tile_boxes, tile_scores):
                    x1, y1, x2, y2 = map(float, box)

                    # Convert tile coordinates into full-image coordinates.
                    x1 += x_offset
                    x2 += x_offset
                    y1 += y_offset
                    y2 += y_offset

                    # Clamp coordinates to image boundaries.
                    x1 = max(0, min(x1, image.width))
                    x2 = max(0, min(x2, image.width))
                    y1 = max(0, min(y1, image.height))
                    y2 = max(0, min(y2, image.height))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    boxes_all.append([x1, y1, x2, y2])
                    scores_all.append(float(score))

            progress.progress(
                (tile_number + 1) / len(tiles),
                text=f"Analyzing tile {tile_number + 1} of {len(tiles)}...",
            )

    finally:
        progress.empty()

    kept_indices = remove_duplicate_boxes(
        boxes_all,
        scores_all,
        iou_threshold=0.45,
    )

    # Draw all retained detections on the original-resolution image.
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for index in kept_indices:
        x1, y1, x2, y2 = boxes_all[index]
        score = scores_all[index]

        draw.rectangle(
            (int(x1), int(y1), int(x2), int(y2)),
            outline="#00FF00",
            width=3,
        )

        draw.text(
            (int(x1) + 2, max(0, int(y1) - 16)),
            f"{score:.2f}",
            fill="#00FF00",
        )

    return len(kept_indices), annotated, len(tiles)


# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------

with st.sidebar:
    st.header("Detection settings")

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.80,
        value=0.15,
        step=0.05,
        help=(
            "Lower values may find more distant heads, but may "
            "also produce false detections."
        ),
    )

    st.caption("Try 0.15, 0.25, and 0.35 to compare results.")

    st.write("**Tiled detection settings**")
    st.write(f"Tile size: {TILE_SIZE} × {TILE_SIZE}")
    st.write(f"Tile overlap: {int(TILE_OVERLAP * 100)}%")


# --------------------------------------------------
# UPLOAD PHOTO
# --------------------------------------------------

uploaded = st.file_uploader(
    "Upload a classroom photo",
    type=["jpg", "jpeg", "png", "webp"],
    help="Maximum upload size: 20 MB.",
)

image = None
digest = None

if uploaded is not None:
    raw_bytes = uploaded.getvalue()

    if len(raw_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        st.error("The file exceeds the 20 MB upload limit.")
    else:
        try:
            image = read_uploaded_image(raw_bytes)
            digest = hashlib.sha256(raw_bytes).hexdigest()

            st.image(
                image,
                caption="Original classroom photo",
                use_container_width=True,
            )

            st.caption(
                f"Resolution: {image.width:,} × {image.height:,} pixels"
            )

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            Image.DecompressionBombError,
        ) as error:
            st.error(f"Could not read this image: {error}")


# --------------------------------------------------
# COUNT BUTTON
# --------------------------------------------------

if st.button(
    "Count heads",
    type="primary",
    use_container_width=True,
    disabled=image is None,
):
    st.session_state.pop("result", None)

    try:
        with st.spinner(
            "Loading the model and analyzing the classroom..."
        ):
            model = load_model()
            start_time = time.perf_counter()

            count, annotated, tile_count = detect_heads_tiled(
                model,
                image,
                confidence,
            )

            elapsed = time.perf_counter() - start_time

            st.session_state["result"] = {
                "digest": digest,
                "count": int(count),
                "confidence": float(confidence),
                "elapsed": float(elapsed),
                "tiles": int(tile_count),
                "image": image_to_jpeg_bytes(annotated),
                "filename": uploaded.name.rsplit(".", 1)[0],
            }

    except Exception as error:
        st.error("Head detection failed.")
        st.exception(error)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

result = st.session_state.get("result")

if result and digest and result["digest"] == digest:
    st.divider()
    st.subheader("Head-count results")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Estimated heads",
        result["count"],
    )

    col2.metric(
        "Confidence threshold",
        f'{result["confidence"]:.2f}',
    )

    col3.metric(
        "Processing time",
        f'{result["elapsed"]:.1f} seconds',
    )

    st.caption(
        f'Analyzed {result["tiles"]} overlapping image tiles.'
    )

    if abs(result["confidence"] - confidence) > 0.001:
        st.warning(
            "The confidence setting has changed. Click Count heads "
            "again to run a new analysis."
        )

    original_column, result_column = st.columns(2)

    with original_column:
        st.image(
            image,
            caption="Original image",
            use_container_width=True,
        )

    with result_column:
        st.image(
            result["image"],
            caption="Detected heads (green boxes)",
            use_container_width=True,
        )

    st.download_button(
        "Download annotated photo",
        data=result["image"],
        file_name=f'{result["filename"]}_head_counted.jpg',
        mime="image/jpeg",
        use_container_width=True,
    )

    st.warning(
        "Please verify the green boxes manually. Overlapping tiles "
        "and duplicate removal reduce some errors, but they cannot "
        "guarantee that every student is detected exactly once."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "This tool provides an AI estimate, not a verified attendance record. "
    "Results can be affected by occlusion, lighting, image quality, "
    "head coverings, and the model's training data."
)

st.caption(
    "Photos are processed by the app server. Do not upload images "
    "unless you have permission to use them."
)
