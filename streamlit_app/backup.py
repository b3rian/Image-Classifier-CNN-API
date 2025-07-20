import streamlit as st
import requests
import io
import base64
from PIL import Image, ImageOps
import numpy as np
import json
import time
import pandas as pd
from typing import List
from datetime import datetime

# Configs
API_URL = "http://127.0.0.1:8000/predict" # API endpoint for image classification
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "webp"] # Supported image formats
MAX_SIZE_MB = 10 # Maximum file size in MB
MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024 # Convert to bytes

# =================== Utility Functions ===================
def compress_image(image: Image.Image, quality: int = 85) -> bytes:
    """Compress image to reduce file size."""
    with io.BytesIO() as output:
        image.save(output, format='JPEG', quality=quality)
        return output.getvalue() 

def create_thumbnail(image: Image.Image, size=(128, 128)) -> str:
    """Create a thumbnail of the image and return it as a base64 string."""
    image.thumbnail(size)
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG", quality=70)
        return base64.b64encode(buffer.getvalue()).decode()

def validate_image(file) -> Image.Image:
    """Validate and open an image file."""
    try:
        if hasattr(file, 'size') and file.size > MAX_SIZE_BYTES:
            st.error(f"File too large (max {MAX_SIZE_MB}MB)")
            return None
        image = Image.open(file)
        image.verify()
        image = Image.open(file)
        return image.convert("RGB")
    except Exception as e:
        st.error(f"Invalid image: {str(e)}")
        return None

def fetch_image_from_url(url: str) -> Image.Image:
    """Fetch an image from a URL and return it as a PIL Image."""
    try:
        with st.spinner("Fetching image from URL..."):
            head_response = requests.head(url, timeout=20, allow_redirects=True)
            if head_response.status_code != 200:
                raise ValueError(f"URL returned {head_response.status_code}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        st.error(f"URL Error: {str(e)}") 
        return None 

def get_image_metadata(img: Image.Image) -> str:
    """Get metadata of the image."""
    return f"Size: {img.size}, Mode: {img.mode}, Format: {img.format}"

def classify_image_with_retry(image: Image.Image, model_name: str, max_retries=2):
    """Classify an image using the specified model with retry logic."""
    img_bytes = compress_image(image) 
    files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
    params = {"model_name": model_name}
    
    for attempt in range(max_retries + 1): 
        try:
            with st.spinner(f"Classifying with {model_name}..."):
                res = requests.post(API_URL, files=files, params=params, timeout=120)
                res.raise_for_status()
                return res.json()
        except requests.exceptions.ConnectionError:
            if attempt == max_retries:
                st.error("⚠️ The model server is currently offline. Please try again later.")
                return None
            time.sleep(1)
        except requests.exceptions.Timeout:
            if attempt == max_retries:
                st.error("⏳ The request to the model server timed out. Please try again.")
                return None
            time.sleep(1)
        except requests.exceptions.HTTPError as e:
            st.error(f"🚫 HTTP error: {e.response.status_code} - {e.response.reason}")
            return None
        except requests.exceptions.RequestException:
            if attempt == max_retries:
                st.error("🚨 An unexpected error occurred while contacting the model server.")
                return None
            time.sleep(1)

def display_predictions(predictions, model_version, inference_time):
    """Display predictions in a user-friendly format."""
    st.subheader(f"Predictions: {model_version}")
    if not predictions:
        st.warning("No predictions above the confidence threshold.")
        return
    df = pd.DataFrame(predictions)
    df = df.set_index("label")

    for pred in predictions:
        st.markdown(f"**{pred['label']}**: {pred['confidence']}%")
        st.progress(pred['confidence'] / 100.0) 

    st.caption(f"Inference time: {inference_time:.2f}s") 

# =================== Main Application ===================
def main():
    """Main function to run the Streamlit app."""
    st.markdown("---")
    st.set_page_config(page_title="Image Classifier", layout="wide", page_icon="🖼️")
    st.title("🖼️ AI Image Classifier")
    st.caption("Powered by Convolutional Neural Networks (CNNs)")

    st.markdown("""
    📌 Upload or capture an image and choose a CNN model to classify it.

    🔍 **How it works**:  
    The selected AI model analyzes your image and returns its best predictions, sorted by confidence.
    """)

    # Initialize session state
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("feedback", {})
    st.session_state.setdefault("model_cache", {})

    # Sidebar controls
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ⚙️ Preferences & Model Selection")
        with st.expander("Advanced Options"):
            num_predictions = st.slider(
                "Number of predictions", 
                1, 10, 3,
                help="""Set how many predictions to display (1-10). 
                Higher values show more alternatives but may include less relevant results."""
        )
            confidence_threshold = st.slider(
                "Confidence threshold (%)", 
                0, 100, 0,
                help="""Minimum confidence percentage (0-100%) required to show a prediction. 
                Increase to filter out low-confidence results."""
        )
            compare_models = st.checkbox(
                "🔁 Compare Models", 
                help="Run both models on the image and compare their predictions."
        )

        model_name = st.selectbox(
            "Select 🧠 AI Model", 
            ["efficientnet", "resnet"], 
            disabled=compare_models,
            help="""Choose a deep learning architecture: 
            • **EfficientNet:** Lightweight and fast (good for mobile/edge devices)
            • **ResNet:** Powerful general-purpose model (best accuracy/speed balance).
            Disabled when 'Compare Models' is active - all models will run simultaneously."""
        )

        st.markdown("---")
        st.subheader("💬 Feedback")

        with st.form("feedback_form_sidebar"):
            history = st.session_state["history"]
            if history:
                selected = st.selectbox("Select image to review", [h["name"] for h in history],
                help="""Choose a previously classified image to provide feedback on. 
                The model's predictions for this image will be shown below for reference.
                Only images with existing classification results appear here.""")
                rating = st.select_slider("Rating (1-5)", options=[1, 2, 3, 4, 5], value=3,
                help="""Rate the model's accuracy for this image:
                1 = Completely wrong • 2 = Mostly incorrect • 3 = Partially correct
                4 = Mostly accurate • 5 = Perfect prediction """)
                selected_item = next((h for h in history if h["name"] == selected), None)
                if selected_item:
                    st.markdown("**Model Predictions:**")
                    for pred in selected_item["predictions"]:
                        st.markdown(f"- {pred['label']}: {pred['confidence']:.1f}%")
                correction = st.text_input("Suggested correction", placeholder="Correct label",
                help="""If the AI's prediction was wrong, please provide:
                • The accurate label for this image
                • Be specific (e.g., 'Golden Retriever' instead of just 'Dog')
                • Use singular nouns where applicable
                Your input helps train better models!""")
                comment = st.text_area("Additional comments", placeholder="Anything else?",
                help="""Share details to improve the model:
                • What features did the AI miss?
                • Was the mistake understandable?
                • Any edge cases we should know about?
    
(Examples: 'The turtle was partially obscured' or 'Confused labrador with golden retriever')""")
            else:
                st.info("No images classified yet.")
                selected = rating = correction = comment = None

            if st.form_submit_button("Submit Feedback", type='primary') and selected:
                st.session_state["feedback"][selected] = {
                    "rating": rating,
                    "predictions": selected_item.get("predictions", []),
                    "correction": correction,
                    "comment": comment,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                st.toast("Feedback saved!", icon="✅")

    # Image input methods
    images = []
    tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "📷 Use Webcam", "🌐 From URL"])
    
    with tab1:
        uploaded_files = st.file_uploader("Upload Image(s)", type=SUPPORTED_FORMATS, accept_multiple_files=True)
        for file in uploaded_files:
            img = validate_image(file)
            if img:
                images.append((img, file.name))

    with tab2:
        try:
            picture = st.camera_input("Capture Image")
            if picture:
                img = validate_image(picture)
                if img:
                    images.append((img, f"webcam_{time.strftime('%Y%m%d_%H%M%S')}.jpg"))
        except Exception:
            st.error("Webcam not supported on this device.")

    with tab3:
        url = st.text_input("Image URL", placeholder="https://example.com/image.jpg")
        col1, col2 = st.columns([3, 1])
        if col1.button("Fetch Image", type='primary') and url:
            img = fetch_image_from_url(url)
            if img:
                images.append((img, f"url_{time.strftime('%Y%m%d_%H%M%S')}.jpg"))
        if col2.button("Clear URL", type='primary'):
            url = ""

    # Classify images
    if images:
        st.subheader("🖼️ Image Preview")
        for idx, (img, name) in enumerate(images):
            with st.expander(f"Image: {name}", expanded=True):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(img, caption=name, use_container_width=True)
                with col2:
                    st.markdown(get_image_metadata(img))
                    if st.button("🚀 Classify Image", key=f"classify_{idx}", type='primary'):
                        models_to_run = ["efficientnet", "resnet"] if compare_models else [model_name]
                        for model in models_to_run:
                            cache_key = f"{name}_{model}"
                            result = st.session_state.model_cache.get(cache_key)
                            if result:
                                st.toast(f"Using cached result for {model}")
                            else:
                                result = classify_image_with_retry(img, model)
                                if result:
                                    st.session_state.model_cache[cache_key] = result

                            if result:
                                preds = [p for p in result['predictions'] if p['confidence'] >= confidence_threshold][:num_predictions]
                                display_predictions(preds, result['model_version'], result['inference_time'])
                                st.session_state.history.append({
                                    "name": name,
                                    "predictions": preds,
                                    "model": result['model_version'],
                                    "time": result.get('timestamp', datetime.now().isoformat()),
                                    "thumbnail": create_thumbnail(img)
                                })

    # Show history
    st.divider()
    st.subheader("📜 Session History")
    if not st.session_state.history:
        st.info("No classification history.")
    else:
      for record in reversed(st.session_state.history[-5:]):
        with st.container(border=True):
            col1, col2 = st.columns([1, 4])
            with col1:
                if "thumbnail" in record:
                    st.image(io.BytesIO(base64.b64decode(record["thumbnail"])))
            with col2:
                st.markdown(f"**{record['name']}**")
                st.caption(f"Model: `{record['model']}` | {record['time']}")
                if record['predictions']:
                    top_pred = record['predictions'][0]
                    st.markdown(f"**Top Prediction**: {top_pred['label']} ({top_pred['confidence']:.1f}%)")
                if record['name'] in st.session_state.feedback:
                    fb = st.session_state.feedback[record['name']]
                    st.markdown(f"Feedback: ⭐{fb['rating']}/5")
                    if fb['correction']:
                        st.markdown(f"*Suggested correction: {fb['correction']}*")

    # Download button
    st.download_button(
        "📥 Download Results as JSON",
        data=json.dumps(st.session_state.history, indent=2),
        file_name="classification_history.json",
        type='primary',
        use_container_width=True
    )

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit")

if __name__ == "__main__":
    main()
