
import tempfile
import requests
import numpy as np
from PIL import Image
from typing import List
import streamlit as st

SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "JPG"}  # Handles .jpg, .jpeg, etc.


def validate_image_format(format_str: str) -> bool:
    """
    Checks if the image format is among the supported formats.

    Args:
        format_str (str): Format string from PIL.Image.format

    Returns:
        bool: True if format is supported
    """
    return format_str.upper() in SUPPORTED_FORMATS


def load_images_from_upload(files: List) -> List[Image.Image]:
    """
    Loads and validates uploaded image files.

    Args:
        files (List): List of file-like objects

    Returns:
        List[Image.Image]: Validated images
    """
    images = []
    for file in files:
        try:
            image = Image.open(file)
            img_format = image.format
            if validate_image_format(img_format):
                image = image.convert("RGB")
                images.append(image)
            else:
                st.warning(f"Unsupported format: {file.name} ({img_format})")
        except Exception:
            st.error(f"Could not load image: {file.name}")
    return images


def load_image_from_url(url: str) -> Image.Image:
    """
    Loads and validates image from a URL.

    Args:
        url (str): Direct image URL

    Returns:
        Image.Image or None: Loaded image or None on error
    """
    try:
        response = requests.get(url, stream=True, timeout=5)
        response.raise_for_status()
        
        # Open image directly from byte stream
        image = Image.open(response.raw)
        img_format = image.format
        if validate_image_format(img_format):
            return image.convert("RGB")
        else:
            st.warning(f"Unsupported image format from URL: {img_format}")
    except Exception as e:
        st.error(f"Failed to load image from URL: {e}")
    return None
