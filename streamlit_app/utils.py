 """
Utility functions for image processing and validation.
"""

import io
import base64
from PIL import Image
import requests
import streamlit as st
from config import MAX_SIZE_BYTES

def compress_image(image: Image.Image, quality: int = 85) -> bytes:
    """Compress an image to JPEG format with specified quality."""
    with io.BytesIO() as output:
        image.save(output, format='JPEG', quality=quality)
        return output.getvalue()

def create_thumbnail(image: Image.Image, size=(128, 128)) -> str:
    """Create a base64-encoded thumbnail of the image."""
    image.thumbnail(size)
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG", quality=70)
        return base64.b64encode(buffer.getvalue()).decode()

def validate_image(file) -> Image.Image:
    """Validate an image file and return a PIL Image object if valid."""
    try:
        if hasattr(file, 'size') and file.size > MAX_SIZE_BYTES:
            st.error(f"File too large (max {MAX_SIZE_MB}MB)")
            return None
        image = Image.open(file)
        image.verify()
        image = Image.open(file)
        return image.convert("RGB")
    except Exception as e:
        st.error(f"Invalid image: {str(e)}")
        return None

def fetch_image_from_url(url: str) -> Image.Image:
    """Fetch an image from a URL and return a PIL Image object."""
    try:
        with st.spinner("Fetching image from URL..."):
            head_response = requests.head(url, timeout=20, allow_redirects=True)
            if head_response.status_code != 200:
                raise ValueError(f"URL returned {head_response.status_code}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        st.error(f"URL Error: {str(e)}")
        return None 

def get_image_metadata(img: Image.Image) -> str:
    """Generate a string with basic image metadata."""
    return f"Size: {img.size}, Mode: {img.mode}, Format: {img.format}"