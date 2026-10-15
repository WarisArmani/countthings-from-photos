
import io
import hashlib
import time

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageDraw, UnidentifiedImageError
from huggingface_hub import hf_hub_download


# =========================================================
# CONFIGURATION
# =========================================================

APP_TITLE = "Classroom Head Counter"

MODEL_REPO = "wesjos/YOLO-head-detection"
MODEL_FILE = "v8s-head.pt"

MAX_UPLOAD_MB = 20
MAX_IMAGE_PIXELS = 25_000_000

TILE_SIZE = 640
TILE_OVERLAP = 0.25
INFERENCE_SIZE = 960

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="👥",
    layout="wide",
)

st.title("Classroom Head Counter")

st.write(
    "Upload a classroom photo to detect and count visible heads. "
    "The app analyzes overlapping sections to help find smaller "
    "heads in the back rows."
)

st.info(
    "This is an estimated count. Please check the marked image "
    "for missed heads and false detections."
)


# =========================================================
# LOAD THE HEAD-SPECIFIC MODEL
# =========================================================

@st.cache_resource(show_spinner=False)
def load_model():
    from ultralytics import YOLO

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
    )

    model = YOLO(model_path)

    # IMPORTANT:
    # Do not assign to model.names.
    # Ultralytics manages this read-only property.
    # Preserve the original class names stored in the model.

    return model


# =========================================================
# IMAGE HELPERS
# =========================================================

def read_image(raw_bytes):
    with Image.open(io.BytesIO(raw_bytes)) as source:
        width, height = source.size

        if width < 1 or height < 1:
            raise ValueError("Invalid image dimensions.")

        if width * height > MAX_IMAGE_PIXELS:
            raise ValueError(
                "The image is too large. Please upload an image "
                "with no more than 25 million pixels."
            )

        return ImageOps.exif_transpose(source).convert("RGB")


def to_jpeg_bytes(image):
    output = io.BytesIO()

    image.save(
        output,
        format="JPEG",
        quality=92,
        optimize=True,
    )

    return output.getvalue()


# =========================================================
# CREATE OVERLAPPING TILES
# =========================================================

def make_tiles(image):
    width, height = image.size

    step = max(
        1,
        int(TILE_SIZE * (1 - TILE_OVERLAP)),
    )

    def positions(length):
        if length <= TILE_SIZE:
            return [0]

        values = list(
            range(
                0,
                length - TILE_SIZE + 1,
                step,
            )
        )

        final_position = length - TILE_SIZE

        if values[-1] != final_position:
            values.append(final_position)

        return values

    tiles = []

    for y in positions(height):
        for x in positions(width):
            right = min(x + TILE_SIZE, width)
            bottom = min(y + TILE_SIZE, height)

            tile = image.crop(
                (x, y, right, bottom)
            )

            tiles.append((x, y, tile))

    return tiles


# =========================================================
# DUPLICATE DETECTION REMOVAL
# =========================================================

def calculate_iou(box, other_boxes):
    x1 = np.maximum(
        box[0],
        other_boxes[:, 0],
    )

    y1 = np.maximum(
        box[1],
        other_boxes[:, 1],
    )

    x2 = np.minimum(
        box[2],
        other_boxes[:, 2],
    )

    y2 = np.minimum(
        box[3],
        other_boxes[:, 3],
    )

    intersection = (
        np.maximum(0, x2 - x1)
        * np.maximum(0, y2 - y1)
    )

    area_box = (
        max(0, box[2] - box[0])
        * max(0, box[3] - box[1])
    )

    area_other = (
        np.maximum(
            0,
            other_boxes[:, 2] - other_boxes[:, 0],
        )
        *
        np.maximum(
            0,
            other_boxes[:, 3] - other_boxes[:, 1],
        )
    )

    union = area_box + area_other - intersection

    return intersection / np.maximum(
        union,
        1e-9,
    )


def remove_duplicates(
    boxes,
    scores,
    iou_threshold=0.45,
):
    if len(boxes) == 0:
        return []

    boxes = np.asarray(
        boxes,
        dtype=np.float32,
    )

    scores = np.asarray(
        scores,
        dtype=np.float32,
    )

    order = np.argsort(scores)[::-1]

    kept = []

    while len(order):
        current = order[0]

        kept.append(int(current))

        if len(order) == 1:
            break

        remaining = order[1:]

        overlaps = calculate_iou(
            boxes[current],
            boxes[remaining],
        )

        order = remaining[
            overlaps < iou_threshold
        ]

    return kept


# =========================================================
# RUN DETECTION ON TILES
# =========================================================

def detect_heads(
    model,
    image,
    confidence,
    progress_bar,
):
    tiles = make_tiles(image)

    all_boxes = []
    all_scores = []

    for tile_index, (
        offset_x,
        offset_y,
        tile,
    ) in enumerate(tiles):

        prediction = model.predict(
            source=np.asarray(tile),
            imgsz=INFERENCE_SIZE,
            conf=float(confidence),
            iou=0.60,
            max_det=500,
            device="cpu",
            verbose=False,
        )[0]

        if (
            prediction.boxes is not None
            and len(prediction.boxes)
        ):
            tile_boxes = (
                prediction.boxes.xyxy.cpu().numpy()
            )

            tile_scores = (
                prediction.boxes.conf.cpu().numpy()
            )

            for box, score in zip(
                tile_boxes,
                tile_scores,
            ):
                x1, y1, x2, y2 = map(
                    float,
                    box,
                )

                # Convert tile coordinates to
                # original-image coordinates.
                x1 += offset_x
                x2 += offset_x

                y1 += offset_y
                y2 += offset_y

                # Keep coordinates inside the image.
                x1 = max(
                    0,
                    min(x1, image.width),
                )

                x2 = max(
                    0,
                    min(x2, image.width),
                )

                y1 = max(
                    0,
                    min(y1, image.height),
                )

                y2 = max(
                    0,
                    min(y2, image.height),
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                all_boxes.append(
                    [x1, y1, x2, y2]
                )

                all_scores.append(
                    float(score)
                )

        progress_bar.progress(
            (tile_index + 1) / len(tiles),
            text=(
                f"Analyzing tile {tile_index + 1} "
                f"of {len(tiles)}..."
            ),
        )

    # Remove duplicate detections created by
    # overlapping tiles.
    kept_indices = remove_duplicates(
        all_boxes,
        all_scores,
        iou_threshold=0.45,
    )

    annotated = image.copy()

    draw = ImageDraw.Draw(annotated)

    for index in kept_indices:
        x1, y1, x2, y2 = all_boxes[index]

        score = all_scores[index]

        draw.rectangle(
            (
                int(x1),
                int(y1),
                int(x2),
                int(y2),
            ),
            outline="#00FF00",
            width=3,
        )

        draw.text(
            (
                int(x1) + 2,
                max(0, int(y1) - 16),
            ),
            f"Head {score:.2f}",
            fill="#00FF00",
        )

    return (
        len(kept_indices),
        annotated,
        len(tiles),
    )


# =========================================================
# SETTINGS
# =========================================================

with st.sidebar:
    st.header("Detection settings")

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.80,
        value=0.30,
        step=0.05,
        help=(
            "Increase the threshold to reduce false detections. "
            "Decrease it if the model misses visible heads."
        ),
    )

    st.caption(
        "Start at 0.30. Compare 0.20, 0.30, and 0.40 "
        "on the same manually counted photo."
    )

    st.write("**Image processing**")

    st.write(
        f"Tile size: {TILE_SIZE} × {TILE_SIZE}"
    )

    st.write(
        f"Tile overlap: {int(TILE_OVERLAP * 100)}%"
    )


# =========================================================
# UPLOAD
# =========================================================

uploaded = st.file_uploader(
    "Upload classroom photo",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
    help="Maximum file size: 20 MB.",
)

image = None
image_digest = None

if uploaded is not None:
    raw_bytes = uploaded.getvalue()

    if len(raw_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(
            "The uploaded file exceeds 20 MB."
        )

    else:
        try:
            image = read_image(raw_bytes)

            image_digest = hashlib.sha256(
                raw_bytes
            ).hexdigest()

            st.image(
                image,
                caption="Original classroom photo",
                use_container_width=True,
            )

            st.caption(
                f"Image resolution: {image.width:,} × "
                f"{image.height:,} pixels"
            )

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            Image.DecompressionBombError,
        ) as error:
            st.error(
                f"Unable to read the image: {error}"
            )


# =========================================================
# COUNT HEADS
# =========================================================

if st.button(
    "Count heads",
    type="primary",
    use_container_width=True,
    disabled=image is None,
):
    st.session_state.pop(
        "head_result",
        None,
    )

    progress = st.progress(
        0,
        text="Starting detection...",
    )

    try:
        with st.spinner(
            "Loading the head-detection model "
            "and analyzing the photo..."
        ):
            model = load_model()

            start_time = time.perf_counter()

            count, annotated, tile_count = detect_heads(
                model=model,
                image=image,
                confidence=confidence,
                progress_bar=progress,
            )

            elapsed = (
                time.perf_counter() - start_time
            )

            st.session_state["head_result"] = {
                "digest": image_digest,
                "count": int(count),
                "confidence": float(confidence),
                "elapsed": float(elapsed),
                "tiles": int(tile_count),
                "annotated": to_jpeg_bytes(
                    annotated
                ),
                "filename": uploaded.name.rsplit(
                    ".",
                    1,
                )[0],
            }

    except Exception as error:
        st.error("Head detection failed.")

        st.exception(error)

    finally:
        progress.empty()


# =========================================================
# SHOW RESULTS
# =========================================================

result = st.session_state.get(
    "head_result"
)

if (
    result is not None
    and image_digest is not None
    and result["digest"] == image_digest
):
    st.divider()

    st.subheader("Detection results")

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
        f'{result["elapsed"]:.1f} sec',
    )

    st.caption(
        f'Image analyzed in {result["tiles"]} '
        f'overlapping tiles.'
    )

    original_col, result_col = st.columns(2)

    with original_col:
        st.image(
            image,
            caption="Original image",
            use_container_width=True,
        )

    with result_col:
        st.image(
            result["annotated"],
            caption="Detected heads (green boxes)",
            use_container_width=True,
        )

    st.download_button(
        "Download annotated photo",
        data=result["annotated"],
        file_name=(
            f'{result["filename"]}_head_counted.jpg'
        ),
        mime="image/jpeg",
        use_container_width=True,
    )

    st.warning(
        "Review the boxes before using this count for "
        "attendance. A box can still be a false detection, "
        "and some heads may be missed even with this "
        "dedicated model."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "This app estimates visible heads, not verified attendance. "
    "Occlusion, head coverings, lighting, and camera angle can "
    "affect accuracy."
)

st.caption(
    "Photos are processed on the app server. Upload only images "
    "you are authorized to use."
)

st.caption(
    "Model license: AGPL-3.0. Review its terms before "
    "commercial deployment."
)
