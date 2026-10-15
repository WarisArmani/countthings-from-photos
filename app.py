# app.py
import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# --- Page config ---
st.set_page_config(
    page_title="Human Head Counter",
    page_icon="👤",
    layout="centered"
)

# --- Load model (cached so it only loads once) ---
@st.cache_resource
def load_model():
    """
    Load the YOLO head-detection model.
    The model file 'v8s-head.pt' must be in the same directory as this script.
    Download it from: https://huggingface.co/Xulee-0806/YOLO-head-detection
    """
    return YOLO("v8s-head.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.error("Make sure 'v8s-head.pt' is in the same folder as this app.py file.")
    st.stop()

# --- Sidebar controls ---
with st.sidebar:
    st.header("⚙️ Settings")
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.05, max_value=0.95, value=0.25, step=0.05,
        help="Lower = more detections (may include false positives)."
    )
    st.info(
        "Model: `v8s-head.pt` (YOLOv8s fine-tuned for head detection). "
        "Fast and accurate for crowd counting."
    )

# --- UI ---
st.title("👤 Human Head Counter")
st.write(
    "Upload a photo and the app will count the number of human heads "
    "it detects using a YOLOv8 model fine-tuned for head detection."
)

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Could not read image: {e}")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_column_width=True)

    if st.button("🔍 Count Heads", use_container_width=True):
        with st.spinner("Detecting heads..."):
            try:
                # Run inference. The head model has only one class (head).
                results = model(image, conf=confidence, imgsz=640, verbose=False)
            except Exception as e:
                st.error(f"Detection failed: {e}")
                st.stop()

            boxes = results[0].boxes
            count = len(boxes)

            # Plot returns a BGR numpy array — convert to RGB for Streamlit
            annotated = results[0].plot()
            if annotated.ndim == 3 and annotated.shape[2] == 3:
                annotated = annotated[:, :, ::-1]

            with col2:
                st.subheader("✅ Detection Result")
                st.metric(label="Heads Detected", value=count)
                st.image(
                    annotated,
                    caption=f"Detected {count} head(s)",
                    use_column_width=True
                )

            # Details table
            with st.expander("Show detection details"):
                if count > 0:
                    rows = []
                    for i, box in enumerate(boxes):
                        xyxy = box.xyxy[0].tolist()
                        conf = float(box.conf)
                        rows.append({
                            "ID": i + 1,
                            "Confidence": f"{conf:.3f}",
                            "Box (x1,y1,x2,y2)": [round(c, 1) for c in xyxy],
                        })
                    st.dataframe(rows, use_container_width=True)
                else:
                    st.info(
                        "No heads detected. Try lowering the confidence "
                        "threshold in the sidebar."
                    )
