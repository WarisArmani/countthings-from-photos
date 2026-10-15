import io
import hashlib
import time

import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps, UnidentifiedImageError

APP_TITLE = "Human Head Counter"
MODEL_REPO = "AmineSam/irail-crowd-counting-yolov8n"
MODEL_FILE = "best.pt"
MAX_UPLOAD_MB = 20
MAX_IMAGE_PIXELS = 25_000_000

st.set_page_config(page_title=APP_TITLE, page_icon="👥", layout="wide")

st.title("👥 Human Head Counter")
st.write(
    "Upload a classroom, meeting, or small-group photo. The app estimates "
    "the number of visible heads and marks detected heads on the image."
)
st.warning(
    "This is an AI estimate, not a verified attendance count. Heads may be "
    "missed when hidden, blurry, distant, or poorly lit."
)

@st.cache_resource(show_spinner=False)
def load_model():
    # Import only when needed to reduce initial app startup work.
    from ultralytics import YOLO

    weights_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
    )
    return YOLO(weights_path)

def open_uploaded_image(raw_bytes):
    with Image.open(io.BytesIO(raw_bytes)) as source:
        width, height = source.size
        if width < 1 or height < 1:
            raise ValueError("The image has invalid dimensions.")
        if width * height > MAX_IMAGE_PIXELS:
            raise ValueError(
                f"Image is too large ({width * height:,} pixels). "
                f"Please use an image with at most {MAX_IMAGE_PIXELS:,} pixels."
            )
        return ImageOps.exif_transpose(source).convert("RGB")

def image_to_jpeg_bytes(image):
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=92, optimize=True)
    return output.getvalue()

with st.sidebar:
    st.header("Detection settings")
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.80,
        value=0.25,
        step=0.05,
        help="Lower values may find more heads but can add false detections.",
    )
    st.caption("Start at 0.25, then compare results with a manually counted photo.")

uploaded = st.file_uploader(
    "Upload a photo",
    type=["jpg", "jpeg", "png", "webp"],
    help=f"Maximum file size: {MAX_UPLOAD_MB} MB.",
)

image = None
raw_bytes = None
image_hash = None

if uploaded is not None:
    raw_bytes = uploaded.getvalue()
    if len(raw_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"File exceeds {MAX_UPLOAD_MB} MB. Please upload a smaller image.")
    else:
        try:
            image = open_uploaded_image(raw_bytes)
            image_hash = hashlib.sha256(raw_bytes).hexdigest()
            st.image(image, caption="Uploaded photo", use_container_width=True)
            st.caption(f"Image dimensions: {image.width:,} × {image.height:,} pixels")
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            st.error(f"Could not open this image: {exc}")

run_count = st.button(
    "Count heads",
    type="primary",
    use_container_width=True,
    disabled=(image is None),
)

if run_count and image is not None and image_hash is not None:
    st.session_state.pop("head_count_result", None)
    try:
        with st.spinner("Loading the model and detecting heads. The first run may take longer..."):
            model = load_model()
            started = time.perf_counter()
            result = model.predict(
                source=np.asarray(image),
                imgsz=832,
                conf=float(confidence),
                iou=0.70,
                max_det=300,
                device="cpu",
                verbose=False,
            )[0]
            elapsed = time.perf_counter() - started

            if result.boxes is None:
                count = 0
            else:
                count = len(result.boxes)

            # Ultralytics plot() returns a BGR image; convert it to RGB for display.
            marked_bgr = result.plot(boxes=True, labels=True, conf=True)
            marked_rgb = np.ascontiguousarray(marked_bgr[:, :, ::-1])
            marked_image = Image.fromarray(marked_rgb)

            st.session_state["head_count_result"] = {
                "image_hash": image_hash,
                "count": int(count),
                "confidence": float(confidence),
                "seconds": float(elapsed),
                "marked_jpeg": image_to_jpeg_bytes(marked_image),
                "filename": uploaded.name.rsplit(".", 1)[0] if uploaded else "photo",
            }
    except Exception as exc:
        st.error("The model could not process this photo.")
        st.exception(exc)

result_data = st.session_state.get("head_count_result")

if result_data and image_hash and result_data.get("image_hash") == image_hash:
    st.divider()
    st.subheader("Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Estimated heads", result_data["count"])
    col2.metric("Confidence threshold", f'{result_data["confidence"]:.2f}')
    col3.metric("Detection time", f'{result_data["seconds"]:.2f} sec')

    if abs(result_data["confidence"] - confidence) > 0.001:
        st.info("You changed the confidence threshold. Click **Count heads** again to recalculate.")

    original_col, marked_col = st.columns(2)
    with original_col:
        st.image(image, caption="Original photo", use_container_width=True)
    with marked_col:
        st.image(
            result_data["marked_jpeg"],
            caption="Detected heads",
            use_container_width=True,
        )

    st.download_button(
        "Download marked photo",
        data=result_data["marked_jpeg"],
        file_name=f'{result_data["filename"]}_head_counted.jpg',
        mime="image/jpeg",
        use_container_width=True,
    )
elif image is not None:
    st.info("Click **Count heads** to analyze the photo.")

st.divider()
st.caption(
    "Photos are processed by this app server. This script does not intentionally "
    "save uploaded photos to a database or permanent file. Do not upload photos "
    "without appropriate authorization."
)
st.caption(
    "Review the head-detection model and Ultralytics licensing terms before "
    "commercial or proprietary use."
)
