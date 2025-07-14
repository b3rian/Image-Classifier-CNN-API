import streamlit as st
import hashlib
import time
import pandas as pd
import uuid
from streamlit_app.api_client import APIClient
from streamlit_app.image_processor import ImageProcessor
from streamlit_app.ui_components import UIComponents
from streamlit_app.cache_manager import CacheManager
from streamlit_app.feedback_logger import FeedbackLogger

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
    if 'cache_cleared' not in st.session_state:
        st.session_state.cache_cleared = False

def process_image_upload(uploaded_file):
    """Handle image processing and classification pipeline"""
    if not ImageProcessor.validate_image(uploaded_file):
        return False
    
    # Use CacheManager's key generation
    file_hash = CacheManager._generate_cache_key(uploaded_file.getvalue())
    st.session_state.last_upload = file_hash
    
    # Check cache (now returns None if expired)
    cached_result = CacheManager.check_cache(file_hash)
    if cached_result:
        st.session_state.predictions = cached_result
        st.info("Loaded from cache")
        return True
    
    # New classification
    with st.spinner("Classifying image..."):
        start_time = time.time()
        try:
            processed_img = ImageProcessor.preprocess_image(uploaded_file)
            predictions = APIClient.classify_image(processed_img)
            st.session_state.predictions = predictions
            
            # Store with timestamp
            CacheManager.cache_prediction(file_hash, predictions)
            
            st.success(f"Done in {time.time()-start_time:.2f}s")
            return True
        except Exception as e:
            st.error(f"Classification failed: {str(e)}")
            return False

def main():
    """Main application entry point"""
    # Page configuration
    st.set_page_config(
        page_title="Image Classifier Pro",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session
    initialize_session()
    
    # Main UI
    st.title("EfficientNetV2L Image Classifier")
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"],
        help="Max 10MB, 224x224 resolution recommended"
    )
    
    # Processing pipeline
    if uploaded_file:
        if process_image_upload(uploaded_file):
            # Display results if successful
            UIComponents.results_view(st.session_state.predictions)
            
            # Feedback system
            if st.session_state.last_upload == CacheManager._generate_cache_key(uploaded_file.getvalue()):
                UIComponents.feedback_system(st.session_state.predictions)
                UIComponents.feedback_summary()
    
    # Sidebar with additional options
    with st.sidebar:
        st.header("Options")
        
        # Cache control
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Cache"):
                CacheManager.clear_cache()
                st.session_state.predictions = None
                st.session_state.cache_cleared = True
                st.rerun()
        
        if st.session_state.cache_cleared:
            st.success("Cache cleared at: " + st.session_state.get('cache_cleared_time', 'Just now'))
            st.session_state.cache_cleared = False
        
        # Results download
        if st.session_state.get('predictions'):
            st.download_button(
                label="Download Results",
                data=convert_to_csv(st.session_state.predictions),
                file_name="classification_results.csv",
                mime="text/csv"
            )
        
        # Cache info
        if 'prediction_cache' in st.session_state:
            st.caption(f"Cache size: {len(st.session_state.prediction_cache)}/{CacheManager.MAX_CACHE_SIZE}")
            st.caption(f"TTL: {CacheManager.CACHE_TTL//3600}h")

def convert_to_csv(predictions):
    """Convert predictions to CSV format"""
    df = pd.DataFrame(predictions)
    return df.to_csv(index=False)

if __name__ == "__main__":
    main()