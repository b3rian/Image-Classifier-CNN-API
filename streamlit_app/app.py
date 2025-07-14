import streamlit as st
from .api_client import APIClient
from .image_processor import ImageProcessor
from .ui_components import UIComponents
from .cache_manager import CacheManager
import hashlib
import time

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Page Config
st.set_page_config(
    page_title="Image Classifier Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)
