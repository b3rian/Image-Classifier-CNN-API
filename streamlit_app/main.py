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

def process_image_upload(uploaded_file):
    """Handle image processing and classification pipeline"""
    if not ImageProcessor.validate_image(uploaded_file):
        return False
    
    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
    st.session_state.last_upload = file_hash
    
    # Check cache
    cached_result = CacheManager.check_cache(file_hash)
    if cached_result:
        st.session_state.predictions = cached_result
        return True
    
    # New classification
    with st.spinner("Classifying image..."):
        start_time = time.time()
        try:
            processed_img = ImageProcessor.preprocess_image(uploaded_file)
            predictions = APIClient.classify_image(processed_img)
            st.session_state.predictions = predictions
            CacheManager.cache_prediction(file_hash, predictions)
            st.success(f"Done in {time.time()-start_time:.2f}s")
            return True
        except Exception as e:
            st.error(f"Classification failed: {str(e)}")
            return False
        
            
             

 
