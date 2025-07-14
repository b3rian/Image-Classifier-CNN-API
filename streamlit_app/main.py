import streamlit as st
from streamlit_app.image_utils import (
    load_images_from_upload,
    load_image_from_url,
)

st.set_page_config(page_title="Image Classifier", layout="wide")

st.title("🧠 Image Classifier Pro")
st.subheader("Upload or Capture Images")

# --- File Upload ---
st.markdown("### 📂 Upload Images")
uploaded_files = st.file_uploader(
    label="Upload one or more images (JPEG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

uploaded_images = load_images_from_upload(uploaded_files) if uploaded_files else []

# --- Webcam Capture ---
st.markdown("---")
st.markdown("### 📸 Capture from Webcam")
if st.button("Capture Image"):
    webcam_image = capture_webcam_image()
    if webcam_image:
        st.image(webcam_image, caption="Webcam Capture", use_column_width=True)
        uploaded_images.append(webcam_image)

# --- URL Input ---
st.markdown("---")
st.markdown("### 🌐 Load from URL")
url = st.text_input("Paste image URL here")
if url:
    url_image = load_image_from_url(url)
    if url_image:
        st.image(url_image, caption="Image from URL", use_container_width=True)
        uploaded_images.append(url_image)

# --- Display Uploaded Images ---
st.markdown("---")
if uploaded_images:
    st.markdown("### 🖼️ Preview Uploaded Images")
    cols = st.columns(min(4, len(uploaded_images)))
    for i, img in enumerate(uploaded_images):
        with cols[i % len(cols)]:
            st.image(img, use_container_width=True)
else:
    st.info("No images to display yet.")

# --- Rotate option ---
rotate_angle = st.slider("🔄 Rotate image (degrees)", -180, 180, 0)

if uploaded_files:
    st.markdown("### 🖼️ Image Preview & Preprocessing")

    for file in uploaded_files:
        image = safe_load_image(file)
        if not image:
            continue

        # Resize large images
        resized = resize_image(image)

        # Rotate image
        if rotate_angle != 0:
            resized = resized.rotate(rotate_angle)

        # Show original vs resized
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original", use_column_width=True)
            st.write(f"📐 Dimensions: {image.size[0]}x{image.size[1]}")
            st.write(f"💾 Size: {get_file_size(image)}")

        with col2:
            # Cropper
            st.write("✂️ Crop (optional)")
            cropped_img = st_cropper(resized, box_color="#FF4B4B", aspect_ratio=None)

            if cropped_img:
                st.image(cropped_img, caption="Cropped & Resized", use_column_width=True)
                st.write(f"📐 Dimensions: {cropped_img.size[0]}x{cropped_img.size[1]}")
                st.write(f"💾 Size: {get_file_size(cropped_img)}")

        st.markdown("---")
else:
    st.info("Upload image(s) to continue.")