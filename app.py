# app.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Human Head Counter",
    page_icon="👤",
    layout="centered"
)

# --- Model Loading ---
# Use a cache to load the model only once.
@st.cache_resource
def load_model():
    """Load the YOLO head detection model."""
    model_path = "v8s-head.pt"
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please download it and place it in the app directory.")
        st.stop()
    return YOLO(model_path)

model = load_model()

# --- Helper Function for Detection ---
def detect_heads(image, conf_threshold=0.25):
    """
    Run head detection on a PIL image.
    Returns the count, the annotated image, and the raw results.
    """
    # Convert PIL image to a NumPy array (RGB) and then to BGR for OpenCV/YOLO
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Run inference
    # The model is a single-class detector for "head".
    results = model(img_bgr, conf=conf_threshold, imgsz=640, verbose=False)

    # Get the number of detections
    head_count = len(results[0].boxes)

    # Plot the results to get an annotated image (BGR)
    annotated_img_bgr = results[0].plot()

    # Convert back to RGB for display in Streamlit
    annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)

    return head_count, annotated_img_rgb, results

# --- Streamlit UI ---
st.title("👤 Human Head Counter")
st.write(
    "Upload an image, and this app will count the number of human heads "
    "it detects using a YOLOv8 model fine-tuned for head detection."
)

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Lower values detect more heads but may include false positives. "
             "Higher values are more strict."
    )
    st.info(
        "Model: `v8s-head.pt` (YOLOv8s fine-tuned for head detection). "
        "It is fast and accurate for most images."
    )

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)

    # Create two columns for a cleaner layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_column_width=True)

    # Button to trigger detection
    if st.button("🔍 Count Heads", use_container_width=True):
        with st.spinner("Detecting heads..."):
            # Run detection
            count, annotated_img, results = detect_heads(image, conf_threshold=confidence)

            # Display results in the second column
            with col2:
                st.subheader("✅ Detection Result")
                st.metric(label="Heads Detected", value=count)

                if count > 0:
                    st.image(
                        annotated_img,
                        caption=f"Detected {count} head(s)",
                        use_column_width=True
                    )
                else:
                    st.info("No heads were detected in this image. "
                            "Try lowering the confidence threshold in the sidebar.")

            # Optional: Show detection details in an expander
            with st.expander("Show detection details"):
                if count > 0:
                    boxes = results[0].boxes
                    data = []
                    for i, box in enumerate(boxes):
                        coords = box.xyxy[0].tolist()
                        conf = float(box.conf)
                        data.append({
                            "ID": i + 1,
                            "Confidence": f"{conf:.3f}",
                            "Bounding Box (x1, y1, x2, y2)": [
                                round(c, 1) for c in coords
                            ]
                        })
                    st.dataframe(data, use_container_width=True)
                else:
                    st.write("No detections to display.")
