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

# ====================== UI COMPONENTS ======================
def image_uploader() -> List[tuple]:
    """Handles all image input methods"""
    images = []
    mode = st.session_state.input_mode
    
    if mode == "Upload":
        files = st.file_uploader(
            "📤 Upload Images", 
            type=SUPPORTED_FORMATS, 
            accept_multiple_files=True
        )
        for file in files if files else []:
            try:
                img = Image.open(file).convert("RGB")
                images.append((file.name, img))
            except Exception as e:
                st.error(f"Invalid image {file.name}: {str(e)}")
                
    elif mode == "Webcam":
        img_file = st.camera_input("📷 Capture Live")
        if img_file:
            try:
                img = Image.open(img_file).convert("RGB")
                images.append(("webcam_capture.jpg", img))
            except Exception as e:
                st.error(f"Camera error: {str(e)}")
            
    elif mode == "URL":
        url = st.text_input("🌐 Image URL", placeholder="https://example.com/image.jpg")
        if url:
            img = fetch_image_from_url(url)
            if img:
                images.append(("url_image.jpg", img))
    
    return images

def display_results():
    """Show classification results with confidence filtering"""
    if not st.session_state.get("api_done", False):
        return
    
    model = st.session_state.get("selected_model", MODEL_OPTIONS[0])
    min_confidence = st.session_state.min_confidence
    
    for idx in range(len(st.session_state.get("uploaded_images", []))):
        key = f"img_{idx}_{model}"
        if key in st.session_state.api_results:
            result = st.session_state.api_results[key]
            if result:
                with st.expander(f"🔎 Result {idx+1}", expanded=True):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(
                            prepare_for_display(st.session_state.uploaded_images[idx][1]),
                            use_column_width=True
                        )
                    with col2:
                        filtered = [
                            p for p in result["predictions"] 
                            if p['confidence'] >= min_confidence
                        ]
                        if filtered:
                            display_prediction_details(filtered)
                        else:
                            st.warning(f"No predictions above {min_confidence}% confidence")

# ====================== MAIN APP ======================
def main():
    # Initialize session and config
    init_session()
    st.set_page_config(
        page_title="AI Image Classifier", 
        layout="wide",
        page_icon="🖼️"
    )
    
    # ===== SIDEBAR CONTROLS =====
    with st.sidebar:
        st.title("⚙️ Settings")
        st.session_state.input_mode = st.radio(
            "Input Method",
            ["Upload", "Webcam", "URL"],
            index=["Upload", "Webcam", "URL"].index(st.session_state.input_mode)
        )
        
        st.session_state.selected_model = st.selectbox(
            "🧠 AI Model", 
            MODEL_OPTIONS,
            index=0
        )
        
        st.session_state.min_confidence = st.slider(
            "🎚️ Minimum Confidence (%)",
            0, 100, st.session_state.min_confidence
        )
        
        st.session_state.compression = st.slider(
            "🗜️ Image Quality (%)",
            50, 100, st.session_state.compression
        )
        
        if st.button("🌙 Toggle Dark Mode"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    
    # ===== MAIN CONTENT =====
    st.title("🖼️ AI Image Classifier")
    
    # Image Upload Section
    uploaded_images = image_uploader()
    if uploaded_images:
        st.session_state.uploaded_images = uploaded_images
        st.subheader("🖌️ Image Previews")
        
        cols = st.columns(min(3, len(uploaded_images)))
        for idx, (name, img) in enumerate(uploaded_images):
            with cols[idx % len(cols)]:
                with st.container(border=True):
                    st.image(
                        prepare_for_display(img),
                        caption=f"{name} ({img.size[0]}x{img.size[1]})"
                    )
        
        if st.button("🚀 Classify Images", type="primary"):
            api_images = [prepare_for_api(img) for _, img in uploaded_images]
            call_api_async(api_images, st.session_state.selected_model)
            st.rerun()
    
    # Results Display
    display_results()

if __name__ == "__main__":
    main()