"""
Main Streamlit application for the AI Image Classifier.
"""

import io
import json
import base64
import time
from datetime import datetime
import streamlit as st
import pandas as pd
from typing import List
from PIL import Image, ImageOps
import numpy as np

# Import from modules
from streamlit_app.config import SUPPORTED_FORMATS, MAX_SIZE_MB
from streamlit_app.utils import (validate_image, fetch_image_from_url, 
                  get_image_metadata, create_thumbnail)
from streamlit_app.api_helpers import classify_image_with_retry
from streamlit_app.ui_components import display_predictions, setup_sidebar

def main():
    """Main application function."""
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

    # Setup sidebar and get settings
    model_name, num_predictions, confidence_threshold, compare_models = setup_sidebar()

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