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