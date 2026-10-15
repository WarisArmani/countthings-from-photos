from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps, UnidentifiedImageError


# ============================================================
# APPLICATION SETTINGS
# ============================================================

APP_TITLE = "Human Head Counter"

MODEL_REPO = "AmineSam/irail-crowd-counting-yolov8n"
MODEL_FILE = "best.pt"

MODEL_IMAGE_SIZE = 832
MODEL_IOU = 0.75
MAX_DETECTIONS = 300

MAX_UPLOAD_MB = 20
MAX_IMAGE_PIXELS = 25_000_000


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="👥",
    layout="wide",
)


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("👥 Human Head Counter")

st.write(
    "Upload a photograph of a classroom, meeting, or small group. "
    "The AI model will estimate the number of visible human heads "
    "and mark detected heads on the photograph."
)

st.caption(
    "The result is an estimate, not an official attendance record. "
    "Hidden, blurry, distant, or poorly lit heads may be missed. "
    "Some objects may also be incorrectly identified as heads."
)


# ============================================================
# LOAD THE AI MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_head_model():
    """
    Download and cache the pretrained head-detection model.

    The Ultralytics import is delayed until the user runs a count,
    helping the app display its interface before loading PyTorch.
    """

    from ultralytics import YOLO

    weights_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
    )

    model = YOLO(weights_path)

    return model


# ============================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================

def get_uploaded_image(upload_bytes: bytes) -> Image.Image:
    """
    Open an uploaded image, correct its orientation, and convert it
    to RGB. Reject images that exceed the configured pixel limit.
    """

    with Image.open(io.BytesIO(upload_bytes)) as source_image:

        width, height = source_image.size

        if width < 1 or height < 1:
            raise ValueError("The image has invalid dimensions.")

        if width * height > MAX_IMAGE_PIXELS:
            raise ValueError(
                f"The image contains {width * height:,} pixels. "
                f"Please upload an image containing no more than "
                f"{MAX_IMAGE_PIXELS:,} pixels."
            )

        corrected_image = (
            ImageOps.exif_transpose(source_image)
            .convert("RGB")
        )

    return corrected_image


def make_jpeg_bytes(image: Image.Image) -> bytes:
    """Convert a PIL image into downloadable JPEG bytes."""

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=92,
        optimize=True,
    )

    return buffer.getvalue()


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

with st.sidebar:

    st.header("Detection settings")

    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.80,
        value=0.25,
        step=0.05,
        help=(
            "Lower values may detect more heads but can create "
            "false detections. Higher values are stricter but "
            "may miss some heads."
        ),
    )

    st.markdown("**Recommended starting point: 0.25**")

    st.caption(
        "The model runs on the CPU. The first analysis may take "
        "longer because the model must be downloaded and loaded."
    )


# ============================================================
# PHOTO UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Choose a photograph",
    type=["jpg", "jpeg", "png", "webp"],
    help=(
        f"Maximum file size: {MAX_UPLOAD_MB} MB. "
        "Supported formats: JPG, PNG, and WEBP."
    ),
)


image: Image.Image | None = None
upload_bytes: bytes | None = None
upload_hash: str | None = None


if uploaded_file is not None:

    upload_bytes = uploaded_file.getvalue()

    if len(upload_bytes) > MAX_UPLOAD_MB * 1024 * 1024:

        st.error(
            f"The file exceeds {MAX_UPLOAD_MB} MB. "
            "Please upload a smaller photograph."
        )

        upload_bytes = None

    else:

        try:

            image = get_uploaded_image(upload_bytes)

            upload_hash = hashlib.sha256(
                upload_bytes
            ).hexdigest()

            st.image(
                image,
                caption="Uploaded photograph",
                use_container_width=True,
            )

            st.caption(
                f"Image dimensions: {image.width:,} × "
                f"{image.height:,} pixels"
            )

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            Image.DecompressionBombError,
        ) as exc:

            st.error(f"Could not read this image: {exc}")

            image = None
            upload_hash = None


# ============================================================
# COUNT HEADS BUTTON
# ============================================================

count_button = st.button(
    "Count heads",
    type="primary",
    disabled=(image is None or upload_hash is None),
    use_container_width=True,
)


# ============================================================
# RUN HEAD DETECTION
# ============================================================

if (
    count_button
    and image is not None
    and upload_hash is not None
):

    # Remove any previous result before starting a new analysis.
    st.session_state.pop("last_result", None)

    try:

        with st.spinner(
            "Loading the AI model and detecting heads..."
        ):

            model = load_head_model()

            start_time = time.perf_counter()

            prediction = model.predict(
                source=np.asarray(image),
                imgsz=MODEL_IMAGE_SIZE,
                conf=float(confidence_threshold),
                iou=MODEL_IOU,
                max_det=MAX_DETECTIONS,
                device="cpu",
                verbose=False,
            )[0]

            elapsed_seconds = (
                time.perf_counter() - start_time
            )

            # Ultralytics returns the plotted image in BGR order.
            annotated_bgr = prediction.plot(
                boxes=True,
                labels=True,
                conf=True,
            )

            # Convert BGR to RGB for correct display in PIL.
            annotated_rgb = np.ascontiguousarray(
                annotated_bgr[:, :, ::-1]
            )

            annotated_image = Image.fromarray(
                annotated_rgb
            )

            # Count the model's detected bounding boxes.
            if prediction.boxes is None:
                detection_count = 0
            else:
                detection_count = len(prediction.boxes)

            # Store the result for Streamlit reruns.
            st.session_state["last_result"] = {
                "upload_hash": upload_hash,
                "count": int(detection_count),
                "confidence": float(confidence_threshold),
                "elapsed_seconds": float(elapsed_seconds),
                "annotated_jpeg": make_jpeg_bytes(
                    annotated_image
                ),
                "image_width": image.width,
                "image_height": image.height,
                "filename": (
                    Path(uploaded_file.name).stem
                    if uploaded_file
                    else "photo"
                ),
            }

    except Exception as exc:

        st.error(
            "The head-detection model could not process "
            "this photograph. Please try again."
        )

        with st.expander("Technical details"):

            st.code(
                f"{type(exc).__name__}: {exc}"
            )


# ============================================================
# DISPLAY RESULTS
# ============================================================

last_result: dict[str, Any] | None = (
    st.session_state.get("last_result")
)


if (
    last_result
    and upload_hash
    and last_result.get("upload_hash") == upload_hash
):

    st.divider()

    st.subheader("Counting result")

    metric_col, detail_col = st.columns([1, 2])

    with metric_col:

        st.metric(
            "Detected heads (estimate)",
            last_result["count"],
        )

    with detail_col:

        st.write(
            "**Confidence threshold:** "
            f"{last_result['confidence']:.2f}"
        )

        st.write(
            "**Processing time:** "
            f"{last_result['elapsed_seconds']:.2f} seconds"
        )

        st.write(
            "**Detection model:** YOLOv8n head detector"
        )

    if abs(
        last_result["confidence"] - confidence_threshold
    ) > 0.001:

        st.info(
            "The confidence threshold has changed since the "
            "last analysis. Click Count heads again to "
            "recalculate the result."
        )

    original_col, detected_col = st.columns(2)

    with original_col:

        st.image(
            image,
            caption="Original photograph",
            use_container_width=True,
        )

    with detected_col:

        st.image(
            last_result["annotated_jpeg"],
            caption="Photograph with detected heads marked",
            use_container_width=True,
        )

    # Provide the marked photograph as a download.
    st.download_button(
        "Download photograph with head detections",
        data=last_result["annotated_jpeg"],
        file_name=(
            f"{last_result['filename']}_head_counted.jpg"
        ),
        mime="image/jpeg",
        use_container_width=True,
    )

elif image is not None:

    st.info(
        "Click Count heads to analyze this photograph."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Privacy: this application processes the photograph "
    "on the app server. This code does not intentionally "
    "save uploaded photographs to a database or permanent file. "
    "Do not upload photos without appropriate authorization."
)

st.caption(
    "Licensing: the selected model lists CC BY-SA 4.0. "
    "Ultralytics offers AGPL-3.0 or Enterprise licensing. "
    "Review the applicable model and software licenses before "
    "commercial or proprietary use."
)
