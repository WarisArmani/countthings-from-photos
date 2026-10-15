# app.py
import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# --- Page config ---
st.set_page_config(page_title="Human Head Counter", page_icon="👤", layout="centered")

# --- Load model (cached so it only loads once) ---
@st.cache_resource
def load_model():
    """
    Load YOLOv8n. Auto-downloads on first run.
    
    To use a head-specific model later:
    1. Upload your .pt file (e.g. 'v8s-head.pt') to your GitHub repo.
    2. Change the line below to: return YOLO("v8s-head.pt")
    """
    return YOLO("yolov8n.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# --- Sidebar controls ---
with st.sidebar:
    st.header("⚙️ Settings")
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.05, max_value=0.95, value=0.25, step=0.05,
        help="Lower = more detections (may include false positives)."
    )
    show_others = st.checkbox("Show all objects (not just people)", value=False)

# --- UI ---
st.title("👤 Human Head Counter")
st.write("Upload a photo and the app will count the people it detects.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Could not read image: {e}")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Original")
        st.image(image, use_column_width=True)

    if st.button("🔍 Count People", use_container_width=True):
        with st.spinner("Detecting..."):
            # Run inference. If not showing others, restrict to class 0 (person).
            try:
                if show_others:
                    results = model(image, conf=confidence, verbose=False)
                else:
                    results = model(image, conf=confidence, classes=[0], verbose=False)
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
                st.subheader("✅ Result")
                st.metric(label="Detected", value=count)
                st.image(annotated, caption=f"{count} detected", use_column_width=True)

            # Details table
            with st.expander("Show detection details"):
                if count > 0:
                    rows = []
                    for i, box in enumerate(boxes):
                        xyxy = box.xyxy[0].tolist()
                        conf = float(box.conf)
                        cls_id = int(box.cls)
                        rows.append({
                            "ID": i + 1,
                            "Confidence": f"{conf:.3f}",
                            "Class": model.names.get(cls_id, str(cls_id)),
                            "Box (x1,y1,x2,y2)": [round(c, 1) for c in xyxy],
                        })
                    st.dataframe(rows, use_container_width=True)
                else:
                    st.info("No objects detected. Try lowering the confidence threshold.")
