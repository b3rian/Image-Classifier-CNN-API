"""
Configuration constants for the Streamlit image classifier app.
"""

# API Configuration
API_URL = "https://b3rian-image-classifier-api.hf.space/predict" 

# Image Processing Configuration
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "webp"]
MAX_SIZE_MB = 10
MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024