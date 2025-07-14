import streamlit as st
import hashlib
import time
import uuid
from .api_client import APIClient
from .image_processor import ImageProcessor
from .ui_components import UIComponents
from .cache_manager import CacheManager
from .feedback_logger import FeedbackLogger

def initialize_session():
    """Initialize all required session state variables"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'show_correction' not in st.session_state:
        st.session_state.show_correction = False
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'last_upload' not in st.session_state:
        st.session_state.last_upload = None

# Page Config
st.set_page_config(
    page_title="Image Classifier Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main UI
def main():
    st.title("EfficientNetV2L Image Classifier")
    
    # File Upload Section
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"],
        help="Max 10MB, 224x224 resolution recommended"
    )
    
    if uploaded_file:
        if ImageProcessor.validate_image(uploaded_file):
            # Generate unique hash for caching
            file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
            
            # Check cache first
            cached_result = CacheManager.check_cache(file_hash)
            if cached_result:
                st.session_state.predictions = cached_result
            else:
                # Process and classify
                with st.spinner("Classifying image..."):
                    start_time = time.time()
                    processed_img = ImageProcessor.preprocess_image(uploaded_file)
                    predictions = APIClient.classify_image(processed_img)
                    st.session_state.predictions = predictions
                    CacheManager.cache_prediction(file_hash, predictions)
                    st.success(f"Done in {time.time()-start_time:.2f}s")
            
            # Display results
            UIComponents.results_view(st.session_state.predictions)
            UIComponents.feedback_system()

if __name__ == "__main__":
    main()
