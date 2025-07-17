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
    with io.BytesIO() as output:
        image.save(output, format='JPEG', quality=quality)
        return output.getvalue()

def validate_image(file) -> Image.Image:
    try:
        image = Image.open(file)
        image.verify()
        image = Image.open(file)
        return image.convert("RGB")
    except Exception:
        return None

def fetch_image_from_url(url: str) -> Image.Image:
    try:
        head_response = requests.head(url, timeout=5, allow_redirects=True)
        if head_response.status_code != 200:
            raise ValueError(f"URL returned {head_response.status_code}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        st.error(f"URL Error: {str(e)}")
        return None 

def get_image_metadata(img: Image.Image) -> str:
    return f"Size: {img.size}, Mode: {img.mode}, Format: {img.format}"

def classify_image(image: Image.Image, model_name: str):
    img_bytes = compress_image(image)
    files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
    params = {"model_name": model_name}
    
    try:
        with st.spinner(f"Classifying with {model_name}..."):
            res = requests.post(API_URL, files=files, params=params, timeout=10)
            res.raise_for_status()
            return res.json()

    except requests.exceptions.ConnectionError:
        st.error("⚠️ The model server is currently offline. Please try again later.")
        return None

    except requests.exceptions.Timeout:
        st.error("⏳ The request to the model server timed out. Please try again.")
        return None

    except requests.exceptions.HTTPError as e:
        st.error(f"🚫 HTTP error: {e.response.status_code} - {e.response.reason}")
        return None

    except requests.exceptions.RequestException:
        st.error("🚨 An unexpected error occurred while contacting the model server.")
        return None

def display_predictions(predictions, model_version, inference_time):
    st.subheader(f"Predictions: `{model_version}`")
    for pred in predictions:
        label = pred['label']
        confidence = pred['confidence']
        st.markdown(f"**{label}**: {confidence}%")
        st.progress(confidence / 100.0)
    st.caption(f"Inference time: `{inference_time}s`")

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

    # ✅ Initialize session state
    if "history" not in st.session_state:
        st.session_state.history = []
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}

    # Sidebar: Preferences, model selection, and feedback
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ⚙️ Preferences & Model Selection")
        compare_models = st.checkbox("🔁 Compare EfficientNet vs ResNet", help="Run both models on the image and compare their predictions.")
        model_name = st.selectbox(
            "Select 🧠 AI Model", 
            ["efficientnet", "resnet"],
            disabled=compare_models,
            help="Choose one model, or toggle comparison mode above."
        )

        theme = st.radio("Theme", ["Light", "Dark"])
        if theme == "Dark":
            st.markdown("""
                <style>
                .stApp { background-color: #0e1117; }
                .stTextInput>div>div>input, .stTextArea>textarea,
                .stSelectbox>div>div>select {
                    background-color: #333 !important;
                    color: white !important;
                }
                .stSlider>div>div>div>div {
                    background-color: #555 !important;
                }
                .st-bb { background-color: transparent; }
                .st-at { background-color: #333; }
                .css-1d391kg { background-color: #0e1117; }
                </style>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("💬 Feedback")

        with st.form("feedback_form_sidebar"):
           history = st.session_state.get("history", [])

           if history:
            selected = st.selectbox(
            "Select image to review",
            [h["name"] for h in history],
            index=0,
            help="Choose the image whose prediction you want to give feedback on."
            )
            rating = st.radio(
            "Accuracy",
            ["👍 Correct", "👎 Incorrect"],
            horizontal=True
            )
            comment = st.text_area(
            "Additional comments",
            placeholder="Any suggestions or issues?",
            help="Optionally share more details—e.g., what the model got wrong or suggestions for improvement."
           )
           else:
            selected = None
            rating = None
            comment = None
            st.info("No images classified yet. Upload and classify an image to leave feedback.")

            submit = st.form_submit_button("Submit Feedback", type='primary')

            if submit and selected:
              st.session_state.feedback[selected] = {
              "rating": rating,
              "comment": comment,
              "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
              st.toast("Feedback saved!", icon="✅")


    # Tabs for input methods
    tab1, tab2, tab3 = st.tabs(["Upload Image", "Use Webcam", "From URL"])
    images = []

    with tab1:
        uploaded_files = st.file_uploader(
            "📤 Upload Image(s)", 
            type=SUPPORTED_FORMATS, 
            accept_multiple_files=True,
            key="file_uploader"
        )
        for file in uploaded_files:
            img = validate_image(file)
            if img:
                images.append((img, file.name))
            else:
                st.warning(f"{file.name} is not a valid image.")

    with tab2:
        try:
            picture = st.camera_input("📷 Capture Live")
            if picture:
                img = validate_image(picture)
                if img:
                    images.append((img, "webcam.jpg"))
        except Exception:
            st.error("Webcam not supported on this device/browser.")

    with tab3:
        url = st.text_input("🌐 Image URL", placeholder="https://example.com/image.jpg" )
        if st.button("Fetch Image", type='primary') and url:
            img = fetch_image_from_url(url)
            if img:
                images.append((img, "url_image.jpg"))
            else:
                st.warning("Could not load image from URL.")

    # Display and classify
    if images:
        st.subheader("🖌️ Image Preview")
        for idx, (img, name) in enumerate(images):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(img, caption=name, use_container_width=True)
            with col2:
                st.markdown(get_image_metadata(img))
                if st.button("🚀 Classify Image", key=f"btn_{idx}", type='primary'):
                    models_to_run = ["efficientnet", "resnet"] if compare_models else [model_name]
                    for model in models_to_run:
                        result = classify_image(img, model)
                        if result:
                            display_predictions(result['predictions'], result['model_version'], result['inference_time'])
                            st.session_state.history.append({
                                "name": name,
                                "predictions": result['predictions'],
                                "model": result['model_version'],
                                "time": result.get('timestamp', datetime.now().isoformat())
                            })

    # Display session history
    st.divider()
    st.subheader("🕒 Session History")
    for record in reversed(st.session_state.history[-3:]):
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

    # Download history
    st.download_button(
        "📥 Save Results",
        data=json.dumps(st.session_state.history, indent=2),
        file_name="history.json",
        type='primary'
    )

    st.markdown("---")
    st.caption("Image Classifier Web App | Built with 🚀 by B3rian")

if __name__ == "__main__":
    main()
