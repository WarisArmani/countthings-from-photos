import io
import hashlib
import time

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from huggingface_hub import hf_hub_download

APP_TITLE = "Human Head Counter"
MODEL_REPO = "AmineSam/irail-crowd-counting-yolov8n"
MODEL_FILE = "best.pt"
MAX_UPLOAD_MB = 20
MAX_IMAGE_PIXELS = 25_000_000

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="👥",
    layout="wide",
)

st.title("Human Head Counter")
st.write(
    "Upload a photo to estimate the number of visible human heads. "
    "Detected heads will be marked on the output image."
)


@st.cache_resource(show_spinner=False)
def load_model():
    from ultralytics import YOLO

    weights = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
    )
    return YOLO(weights)


def read_image(data):
    with Image.open(io.BytesIO(data)) as source:
        width, height = source.size

        if width * height > MAX_IMAGE_PIXELS:
            raise ValueError(
                "Image is too large. Use an image under 25 million pixels."
            )

        return ImageOps.exif_transpose(source).convert("RGB")


def image_bytes(image):
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=92)
    return output.getvalue()


with st.sidebar:
    st.header("Settings")
    confidence = st.slider(
        "Detection confidence",
        min_value=0.05,
        max_value=0.80,
        value=0.25,
        step=0.05,
    )
    st.caption(
        "Lower confidence may detect more heads but can add false detections."
    )

uploaded = st.file_uploader(
    "Upload a photo",
    type=["jpg", "jpeg", "png", "webp"],
)

image = None
data = None
digest = None

if uploaded is not None:
    data = uploaded.getvalue()

    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        st.error("The file is larger than 20 MB.")
    else:
        try:
            image = read_image(data)
            digest = hashlib.sha256(data).hexdigest()
            st.image(
                image,
                caption="Uploaded photo",
                use_container_width=True,
            )
        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            Image.DecompressionBombError,
        ) as error:
            st.error(f"Unable to open image: {error}")

if st.button(
    "Count heads",
    type="primary",
    disabled=image is None,
    use_container_width=True,
):
    st.session_state.pop("result", None)

    try:
        with st.spinner(
            "Loading model and analyzing the image. "
            "The first run can take several minutes."
        ):
            model = load_model()
            start = time.perf_counter()

            prediction = model.predict(
                source=np.asarray(image),
                imgsz=832,
                conf=confidence,
                iou=0.70,
                max_det=300,
                device="cpu",
                verbose=False,
            )[0]

            elapsed = time.perf_counter() - start

            count = (
                0
                if prediction.boxes is None
                else len(prediction.boxes)
            )

            # Ultralytics returns an annotated BGR image.
            annotated_bgr = prediction.plot()
            annotated_rgb = np.ascontiguousarray(
                annotated_bgr[:, :, ::-1]
            )
            annotated = Image.fromarray(annotated_rgb)

            st.session_state["result"] = {
                "digest": digest,
                "count": int(count),
                "confidence": float(confidence),
                "elapsed": elapsed,
                "image": image_bytes(annotated),
                "filename": uploaded.name.rsplit(".", 1)[0],
            }

    except Exception as error:
        st.error("Head detection failed.")
        st.exception(error)

result = st.session_state.get("result")

if result and digest and result["digest"] == digest:
    st.divider()
    st.subheader("Detection results")

    first, second, third = st.columns(3)

    first.metric("Estimated heads", result["count"])
    second.metric("Confidence", f'{result["confidence"]:.2f}')
    third.metric("Processing time", f'{result["elapsed"]:.2f} seconds')

    if abs(result["confidence"] - confidence) > 0.001:
        st.info("Click Count heads again to apply the new confidence setting.")

    left, right = st.columns(2)

    with left:
        st.image(image, caption="Original", use_container_width=True)

    with right:
        st.image(
            result["image"],
            caption="Detected heads",
            use_container_width=True,
        )

    st.download_button(
        "Download annotated photo",
        data=result["image"],
        file_name=f'{result["filename"]}_counted.jpg',
        mime="image/jpeg",
        use_container_width=True,
    )

st.divider()
st.caption(
    "The result is an estimate and may miss or incorrectly detect heads. "
    "Uploaded photos are processed by the app server. Do not upload images "
    "unless you have permission to use them."
)
