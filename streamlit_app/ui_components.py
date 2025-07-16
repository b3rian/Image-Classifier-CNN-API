import streamlit as st
import requests
import io
import base64
from PIL import Image, ImageOps
import numpy as np
import json
import time
from typing import List
from datetime import datetime

# =================== CONFIG ===================
API_URL = "http://127.0.0.1:8000/predict"
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "webp"]

# =================== UTILITY FUNCTIONS ===================
def compress_image(image: Image.Image, quality: int = 85) -> bytes:
    """Compress the image and return as bytes."""
    with io.BytesIO() as output:
        image.save(output, format='JPEG', quality=quality)
        return output.getvalue()

def validate_image(file) -> Image.Image:
    """Validate and open image file."""
    try:
        image = Image.open(file)
        image.verify()  # Check for corruption
        image = Image.open(file)  # Reopen after verify
        return image.convert("RGB")
    except Exception:
        return None

def fetch_image_from_url(url: str) -> Image.Image:
    """Download image from URL and return PIL image."""
    try:
        response = requests.get(url, timeout=5)
        img = Image.open(io.BytesIO(response.content))
        return img.convert("RGB")
    except Exception:
        return None

def get_image_metadata(img: Image.Image) -> str:
    """Return image metadata for display."""
    return f"Size: {img.size}, Mode: {img.mode}, Format: {img.format}"

def classify_image(image: Image.Image, model_name: str):
    """Send image to backend API and get predictions."""
    img_bytes = compress_image(image)
    files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
    params = {"model_name": model_name}

    try:
        with st.spinner("Classifying image..."):
            res = requests.post(API_URL, files=files, params=params, timeout=120)
            res.raise_for_status()
            return res.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

def display_predictions(predictions, model_version, inference_time):
    """Display top predictions with visual confidence."""
    st.subheader("Predictions")
    for pred in predictions:
        label = pred['label']
        confidence = pred['confidence']
        st.markdown(f"**{label}**: {confidence}%")
        st.progress(confidence / 100.0)

    st.caption(f"Model: `{model_version}` | Inference time: `{inference_time}s`")

# =================== MAIN APP ===================
def main():
    st.set_page_config(page_title="Image Classifier", layout="wide", page_icon="🖼️")
    st.title("🖼️ AI Image Classifier")
    st.markdown("""
    Select or capture an image and choose a model to classify.

    🔍 **How it works**:  
    Once you upload an image, the selected AI model analyzes it and returns the top 3 most probable classifications.  
    These predictions are sorted by confidence level — how sure the model is about each label — allowing you to see not only the most likely label but also alternative possibilities.

    📌 *Why top 3?*  
    AI models often identify features shared across multiple categories. Showing the top 3 helps highlight cases where multiple labels are nearly probable, which can be helpful for fine-grained tasks or ambiguous images.
    """)


    # Model selection
    model_name = st.selectbox("Select Model", ["efficientnet", "resnet"])

    # Tabs for multiple input methods
    tab1, tab2, tab3 = st.tabs(["Upload Image", "Use Webcam", "From URL"])
    images = []

    with tab1:
        uploaded_files = st.file_uploader(
            "Upload Image(s)", type=SUPPORTED_FORMATS, accept_multiple_files=True
        )
        for file in uploaded_files:
            img = validate_image(file)
            if img:
                images.append((img, file.name))
            else:
                st.warning(f"{file.name} is not a valid image.")

    with tab2:
        try:
            picture = st.camera_input("Capture from webcam")
            if picture:
                img = validate_image(picture)
                if img:
                    images.append((img, "webcam.jpg"))
        except Exception:
            st.error("Webcam not supported on this device/browser.")

    with tab3:
        url = st.text_input("Paste image URL")
        if st.button("Fetch Image", type='primary') and url:
            img = fetch_image_from_url(url)
            if img:
                images.append((img, "url_image.jpg"))
            else:
                st.warning("Could not load image from URL.")

    # ✅ Initialize session state variables
    if "history" not in st.session_state:
        st.session_state.history = []

    if "feedback" not in st.session_state:
        st.session_state.feedback = {}

    if images:
        st.subheader("🖌️ Image Preview")
        for idx, (img, name) in enumerate(images):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(img, caption=name,  use_container_width=True)
            with col2:
                st.markdown(get_image_metadata(img))
                if st.button("🚀 Classify Image", key=f"btn_{idx}", type='primary'):
                    result = classify_image(img, model_name)
                    if result:
                        display_predictions(
                            result['predictions'],
                            result['model_version'],
                            result['inference_time']
                        )
                        st.session_state.history.append({
                            "name": name,
                            "predictions": result['predictions'],
                            "model": result['model_version'],
                            "time": result.get('timestamp', datetime.now().isoformat())
                        })

    # Display history
    st.divider()
    st.subheader("🕒 Session History")

    if 'history' in st.session_state and st.session_state.history:
      for record in reversed(st.session_state.history[-3:]):  # Show only the last 3
        with st.container(border=True):
            col1, col2 = st.columns([1, 2])
            with col1:
                if "thumbnail" in record:
                    st.image(io.BytesIO(base64.b64decode(record["thumbnail"])), width=60)
                else:
                    st.text("🖼️ No preview")
            with col2:
                st.caption(f"{record['name']} | Model: `{record['model']}`")
                st.write(f"Top: {record['predictions'][0]['label']}")
                st.caption(f"⏱️ {record['time']}")

    st.download_button(
        "📥 Save Results",
        data=json.dumps(st.session_state.history, indent=2),
        file_name="history.json",
        type='primary'
    )

    # Sidebar: Preferences and Feedback
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Preferences")
    theme = st.sidebar.radio("Theme", ["Light", "Dark"])
    if theme == "Dark":
        st.markdown("""
            <style>
            body { background-color: #1e1e1e; color: white; }
            </style>
        """, unsafe_allow_html=True)

    # Feedback System
    if st.session_state.get("history"):
        st.divider()
        st.subheader("💬 Feedback")
        with st.form("feedback_form"):
            selected = st.selectbox(
                "Select image to review",
                [h["name"] for h in st.session_state.history],
                index=0
            )
            rating = st.radio(
                "Accuracy",
                ["👍 Correct", "👎 Incorrect"],
                horizontal=True
            )
            comment = st.text_area("Additional comments")
            
            if st.form_submit_button("Submit Feedback"):
                st.session_state.feedback[selected] = {
                    "rating": rating,
                    "comment": comment,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                st.toast("Feedback saved!", icon="✅")

if __name__ == "__main__":
    main()

st.markdown("---")
st.caption("Image Classifier Web App | Built with 🚀 by B3rian")  