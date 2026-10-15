)

image: Image.Image | None = None
upload_bytes: bytes | None = None
upload_hash: str | None = None

if uploaded_file is not None:
    upload_bytes = uploaded_file.getvalue()
    if len(upload_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"The file is larger than {MAX_UPLOAD_MB} MB. Please upload a smaller photo.")
        upload_bytes = None
    else:
        try:
            image = get_uploaded_image(upload_bytes)
            upload_hash = hashlib.sha256(upload_bytes).hexdigest()
            st.image(image, caption="Uploaded photo", use_container_width=True)
            st.caption(f"Image dimensions: {image.width:,} × {image.height:,} pixels")
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            st.error(f"Could not read this image: {exc}")
            image = None
            upload_hash = None

count_button = st.button(
    "Count heads",
    type="primary",
    disabled=(image is None or upload_hash is None),
    use_container_width=True,
)

if count_button and image is not None and upload_hash is not None:
    # Avoid displaying an older result if this attempt fails.
    st.session_state.pop("last_result", None)
    try:
        with st.spinner("Loading the AI model and detecting heads..."):
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
            elapsed_seconds = time.perf_counter() - start_time

            # Ultralytics returns an annotated image in BGR channel order.
            annotated_bgr = prediction.plot(
                boxes=True,
                labels=True,
                conf=True,
            )
            annotated_rgb = np.ascontiguousarray(annotated_bgr[:, :, ::-1])
            annotated_image = Image.fromarray(annotated_rgb)

            detection_count = 0 if prediction.boxes is None else len(prediction.boxes)
            st.session_state["last_result"] = {
                "upload_hash": upload_hash,
                "count": int(detection_count),
                "confidence": float(confidence_threshold),
                "elapsed_seconds": float(elapsed_seconds),
                "annotated_jpeg": make_jpeg_bytes(annotated_image),
                "image_width": image.width,
                "image_height": image.height,
                "filename": Path(uploaded_file.name).stem if uploaded_file else "photo",
            }
    except Exception as exc:
        st.error(
            "The head-detection model could not process this photo. "
            "Please try again. If the problem continues, check the app logs in Streamlit Community Cloud."
        )
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}")

last_result: dict[str, Any] | None = st.session_state.get("last_result")

if last_result and upload_hash and last_result.get("upload_hash") == upload_hash:
    st.divider()
    st.subheader("Counting result")
    metric_col, detail_col = st.columns([1, 2])
    with metric_col:
        st.metric("Detected heads (estimate)", last_result["count"])
    with detail_col:
        st.write(f"**Confidence threshold used:** {last_result['confidence']:.2f}")
        st.write(f"**Model processing time:** {last_result['elapsed_seconds']:.2f} seconds")
        st.write(f"**Model:** YOLOv8n head detector")

    if abs(last_result["confidence"] - confidence_threshold) > 0.001:
        st.info(
            "You changed the confidence threshold after the last run. "
            "Click **Count heads** again to recalculate with the new setting."
        )

    left_col, right_col = st.columns(2)
    with left_col:
        st.image(image, caption="Original photo", use_container_width=True)
    with right_col:
        st.image(
            last_result["annotated_jpeg"],
            caption="Detected heads marked by the model",
            use_container_width=True,
        )

    st.download_button(
        "Download photo with head detections",
        data=last_result["annotated_jpeg"],
        file_name=f"{last_result['filename']}_head_counted.jpg",
        mime="image/jpeg",
        use_container_width=True,
    )
elif image is not None:
    st.info("Click **Count heads** to analyze this photo.")

st.divider()
st.caption(
    "Privacy note: this app processes the photo on the app server while you use it. "
    "This code does not intentionally save uploaded photos to a database or permanent file. "
    "Do not upload photos you are not authorized to process."
)
st.caption(
    "Licensing notice: the selected model lists CC BY-SA 4.0, and Ultralytics offers AGPL-3.0 or Enterprise licensing. "
    "Review both sets of terms before commercial or proprietary use."
)
