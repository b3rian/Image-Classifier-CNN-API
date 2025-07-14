import tempfile
import requests
import numpy as np
from PIL import Image
from typing import List
import streamlit as st


SUPPORTED_FORMATS = ["JPEG", "PNG", "WEBP"]

def validate_image_format(image: Image.Image) -> bool:
    """
    Validates if the image format is supported.

    Args:
        image (Image.Image): PIL Image

    Returns:
        bool: True if supported, False otherwise
    """
    return image.format in SUPPORTED_FORMATS

def load_images_from_upload(files: List) -> List[Image.Image]:
    """
    Loads and validates uploaded images.

    Args:
        files (List): List of uploaded file-like objects

    Returns:
        List[Image.Image]: Valid images
    """
    images = []
    for file in files:
        try:
            image = Image.open(file).convert("RGB")
            if validate_image_format(image):
                images.append(image)
            else:
                st.warning(f"Unsupported format: {file.name}")
        except Exception:
            st.error(f"Could not load image: {file.name}")
    return images

def load_image_from_url(url: str) -> Image.Image:
    """
    Loads and validates image from a URL.

    Args:
        url (str): Image URL

    Returns:
        Image.Image: Validated image or None
    """
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        image = Image.open(tempfile.NamedTemporaryFile(delete=False, suffix=".jpg"))
        image = Image.open(response.content).convert("RGB")
        if validate_image_format(image):
            return image
        st.warning("Unsupported image format from URL.")
    except Exception as e:
        st.error(f"Failed to load image from URL: {e}")
    return None
