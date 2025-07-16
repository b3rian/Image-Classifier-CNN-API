import streamlit as st
import requests
import io
import base64
from PIL import Image
from typing import List, TypedDict, Optional
import time

# ====================== CONSTANTS & CONFIG ======================
API_URL = "http://127.0.0.1:8000/predict"  # FastAPI endpoint
SUPPORTED_FORMATS = ["png", "jpg", "jpeg", "bmp"]
MODEL_OPTIONS = ["ResNet", "Efficientnet"]
MAX_IMAGE_DIM = 480  # For resizing
DEFAULT_COMPRESSION = 85  # JPEG quality

class Prediction(TypedDict):
    label: str
    confidence: float

class ApiResponse(TypedDict):
    predictions: List[Prediction]
    model_version: str
    inference_time: float

# ====================== SESSION INITIALIZATION ======================
def init_session():
    """Initialize all session state variables"""
    session_defaults = {
        "history": [],
        "dark_mode": False,
        "api_results": {},
        "feedback": {},
        "input_mode": "Upload",
        "min_confidence": 30,
        "compression": DEFAULT_COMPRESSION
    }
    
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ====================== CORE FUNCTIONS ======================
def compress_image(image: Image.Image, quality: int = None) -> bytes:
    """Compress image with adjustable quality"""
    if quality is None:
        quality = st.session_state.get("compression", DEFAULT_COMPRESSION)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def validate_image(file) -> bool:
    """Verify image integrity"""
    try:
        Image.open(file).verify()
        return True
    except Exception as e:
        st.error(f"Invalid image: {str(e)}")
        return False

def resize_image(image: Image.Image, max_dim: int = MAX_IMAGE_DIM) -> Image.Image:
    """Maintain aspect ratio while resizing"""
    image.thumbnail((max_dim, max_dim))
    return image

def fetch_image_from_url(url: str) -> Optional[Image.Image]:
    """Fetch with URL validation and timeout"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        st.error(f"URL Error: {str(e)}")
        return None

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
        st.error("API timed out. Server may be busy.")
        return None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        if hasattr(e, "response") and e.response:
            st.error(f"API Response: {e.response.text}")
        return None

# ====================== UI COMPONENTS ======================
def display_predictions(predictions: List[Prediction]):
    """Interactive results display with confidence filter"""
    if not predictions:
        st.warning("No predictions returned")
        return
    
    min_confidence = st.session_state.get("min_confidence", 30)
    filtered = [p for p in predictions if p['confidence'] >= min_confidence]
    
    if not filtered:
        st.warning(f"No predictions above {min_confidence}% confidence")
        return
    
    top = filtered[0]
    with st.container(border=True):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Top Prediction", 
                     f"{top['label']}", 
                     f"{top['confidence']:.1f}%")
        with col2:
            st.progress(int(top['confidence']), 
                       text=f"Confidence: {top['confidence']:.1f}%")
    
    with st.expander("🔍 Detailed Predictions"):
        for p in filtered:
            st.markdown(
                f"`{p['label']:30s}` | "
                f"`{p['confidence']:5.1f}%` | "
                f"{'█' * int(p['confidence']/10)}"
            )

def image_uploader() -> List[tuple]:
    """Handles all image input methods"""
    images = []
    mode = st.session_state.input_mode
    
    if mode == "Upload":
        files = st.file_uploader(
            "📤 Upload Images", 
            type=SUPPORTED_FORMATS, 
            accept_multiple_files=True,
            key="file_uploader"
        )
        for file in files if files else []:
            if validate_image(file):
                img = Image.open(file).convert("RGB")
                img = resize_image(img)
                images.append((file.name, img))
                
    elif mode == "Webcam":
        img_file = st.camera_input("📷 Capture Live")
        if img_file:
            img = Image.open(img_file).convert("RGB")
            img = resize_image(img)
            images.append(("webcam_capture.jpg", img))
            
    elif mode == "URL":
        url = st.text_input("🌐 Image URL", placeholder="https://example.com/image.jpg")
        if url:
            img = fetch_image_from_url(url)
            if img:
                img = resize_image(img)
                images.append(("url_image.jpg", img))
    
    return images

def save_to_history(name: str, prediction: List[Prediction], img_bytes: bytes):
    """Store results with thumbnail"""
    thumb = Image.open(io.BytesIO(img_bytes))
    thumb.thumbnail((100, 100))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG")
    
    entry = {
        "name": name,
        "predictions": prediction,
        "thumbnail": base64.b64encode(buf.getvalue()).decode(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.history.append(entry)

def apply_theme():
    """Apply dark/light theme based on session state"""
    if st.session_state.get("dark_mode", False):
        st.markdown("""
            <style>
            .stApp { background-color: #0e1117; }
            .stTextInput>div>div>input, .stTextArea>textarea,
            .stSelectbox>div>div>select {
                background-color: #333 !important;
                color: white !important;
            }
            </style>
        """, unsafe_allow_html=True)

# ====================== MAIN APP ======================
def main():
    init_session()
    st.set_page_config(
        page_title="AI Image Classifier", 
        layout="wide",
        page_icon="🖼️"
    )
    apply_theme()
    
    # ===== SIDEBAR =====
    with st.sidebar:
        st.title("⚙️ Settings")
        st.session_state.input_mode = st.radio(
            "Input Method",
            ["Upload", "Webcam", "URL"],
            index=["Upload", "Webcam", "URL"].index(st.session_state.input_mode)
        )
        
        model = st.selectbox("🧠 AI Model", MODEL_OPTIONS, index=0)
        
        st.session_state.min_confidence = st.slider(
            "🎚️ Minimum Confidence (%)",
            0, 100, st.session_state.min_confidence
        )
        
        st.session_state.compression = st.slider(
            "🗜️ Image Compression (%)",
            50, 100, st.session_state.compression
        )
        
        if st.button("🌙 Toggle Dark Mode"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
        
        st.divider()
        st.subheader("🕒 History")
        if st.session_state.history:
            for entry in reversed(st.session_state.history[-3:]):
                with st.container(border=True):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(io.BytesIO(base64.b64decode(entry["thumbnail"])), 
                                width=60)
                    with col2:
                        st.caption(f"{entry['name']}")
                        st.write(f"Top: {entry['predictions'][0]['label']}")
                        st.caption(f"⏱️ {entry['timestamp']}")
    
    # ===== MAIN CONTENT =====
    st.title("🖼️ Smart Image Classifier")
    
    # Image Input Section
    images = image_uploader()
    
    # Image Preview and Adjustment
    if images:
        st.subheader("🖌️ Image Preview")
        cols = st.columns(min(3, len(images)))
        for idx, (name, img) in enumerate(images):
            with cols[idx % len(cols)]:
                with st.container(border=True):
                    st.image(img, caption=name)
                    rotate = st.slider(
                        f"Rotate {name}",
                        0, 360, 0,
                        key=f"rotate_{idx}"
                    )
                    if rotate:
                        images[idx] = (name, img.rotate(rotate))
    
        # Classification Button
        if st.button("🚀 Classify Images", type="primary"):
            st.session_state.api_results = {}  # Clear previous results
            
            with st.spinner("Classifying images..."):
                for idx, (name, img) in enumerate(images):
                    img_bytes = compress_image(img)
                    result = call_api(img_bytes, model)
                    
                    if result:
                        key = f"img_{idx}_{model}"
                        st.session_state.api_results[key] = result
                        save_to_history(name, result["predictions"], img_bytes)
    
    # Display Results
    if st.session_state.get("api_results"):
        st.subheader("📊 Results")
        for idx, (name, img) in enumerate(images):
            key = f"img_{idx}_{model}"
            if key in st.session_state.api_results:
                result = st.session_state.api_results[key]
                with st.expander(f"🔎 {name}", expanded=True):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(img, use_column_width=True)
                    with col2:
                        display_predictions(result["predictions"])

    # Feedback System
    if st.session_state.get("history"):
        st.divider()
        with st.form("feedback_form"):
            st.subheader("💬 Feedback")
            selected = st.selectbox(
                "Select image to review",
                [h["name"] for h in st.session_state.history]
            )
            rating = st.radio("Accuracy", ["👍 Correct", "👎 Incorrect"], horizontal=True)
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
    st.caption("Image Classifier Web App | Built with Streamlit")