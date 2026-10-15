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
