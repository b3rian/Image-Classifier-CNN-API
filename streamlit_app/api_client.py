import streamlit as st
import requests
import io
import base64
import json
from PIL import Image
from typing import List, TypedDict, Optional
from threading import Thread
import time

# ====================== CONSTANTS & CONFIG ======================
API_URL = "http://localhost:8000/predict"
SUPPORTED_FORMATS = ["png", "jpg", "jpeg", "bmp"]
MODEL_OPTIONS = ["ResNet50", "ViT", "MobileNet"]
MAX_DISPLAY_DIM = 800  # For UI previews
MAX_API_DIM = 512      # For API processing
DEFAULT_COMPRESSION = 85

class Prediction(TypedDict):
    label: str
    confidence: float

class ApiResponse(TypedDict):
    predictions: List[Prediction]
    model_version: str
    inference_time: float

# ====================== IMAGE PROCESSING ======================
def resize_image(image: Image.Image, max_dim: int) -> Image.Image:
    """Maintain aspect ratio while resizing"""
    img = image.copy()
    img.thumbnail((max_dim, max_dim))
    return img

def prepare_for_display(image: Image.Image) -> Image.Image:
    """Optimize image for UI preview"""
    return resize_image(image, MAX_DISPLAY_DIM)

def prepare_for_api(image: Image.Image) -> bytes:
    """Optimize image for API transmission"""
    processed = resize_image(image, MAX_API_DIM)
    return compress_image(processed)

def compress_image(image: Image.Image, quality: int = None) -> bytes:
    """Convert image to optimized JPEG bytes"""
    if quality is None:
        quality = st.session_state.get("compression", DEFAULT_COMPRESSION)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

# ====================== SESSION MANAGEMENT ======================
def init_session():
    """Initialize all session state variables"""
    session_defaults = {
        "history": [],
        "dark_mode": False,
        "api_results": {},
        "feedback": {},
        "input_mode": "Upload",
        "api_done": False,
        "min_confidence": 30,
        "compression": DEFAULT_COMPRESSION
    }
    
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ====================== API COMMUNICATION ======================
def call_api(image_bytes: bytes, model_name: str) -> Optional[ApiResponse]:
    """Handle API calls with retry logic"""
    try:
        response = requests.post(
            API_URL,
            files={"file": image_bytes},
            params={"model": model_name},
            timeout=15
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.warning("API timed out. Server may be busy.")
        return None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        if hasattr(e, "response") and e.response:
            st.json(e.response.json())
        return None
    
def call_api_async(images: List[bytes], model_name: str):
    """Non-blocking API call with progress"""
    st.session_state.api_results = {}
    st.session_state.api_done = False
    
    def worker():
        for idx, img_bytes in enumerate(images):
            key = f"img_{idx}_{model_name}"
            with st.spinner(f"Processing image {idx+1}/{len(images)}..."):
                result = call_api(img_bytes, model_name)
                st.session_state.api_results[key] = result
                time.sleep(0.5)
        st.session_state.api_done = True
    
    Thread(target=worker).start()