"""
API helper functions for communicating with the classification service.
"""
from PIL import Image, ImageOps
import time
import requests
import streamlit as st
from streamlit_app.utils import compress_image
from streamlit_app.config import API_URL

def classify_image_with_retry(image: Image.Image, model_name: str, max_retries=2):
    """
    Classify an image with retry logic for handling temporary failures.
    
    Args:
        image: PIL Image object to classify
        model_name: Name of the model to use
        max_retries: Maximum number of retry attempts
        
    Returns:
        dict: Classification results or None if failed
    """
    img_bytes = compress_image(image)
    files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
    params = {"model_name": model_name}
    
    for attempt in range(max_retries + 1):
        # Ensure the image is in RGB format
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