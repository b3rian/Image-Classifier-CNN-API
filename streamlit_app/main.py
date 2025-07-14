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
