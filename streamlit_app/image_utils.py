
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

MAX_DIM = 512  # Max width/height for resizing

def resize_image(image: Image.Image, max_dim: int = MAX_DIM) -> Image.Image:
    """
    Resize image to a maximum dimension while keeping aspect ratio.

    Args:
        image (PIL.Image): Input image
        max_dim (int): Maximum width/height

    Returns:
        PIL.Image: Resized image
    """
    w, h = image.size
    if max(w, h) <= max_dim:
        return image

    scale = max_dim / float(max(w, h))
    new_size = (int(w * scale), int(h * scale))
    return image.resize(new_size, Image.ANTIALIAS)

def get_file_size(image: Image.Image) -> str:
    """
    Estimate file size of an image in memory.

    Args:
        image (PIL.Image): Input image

    Returns:
        str: Size in KB/MB
    """
    with io.BytesIO() as output:
        image.save(output, format="JPEG")
        size_kb = len(output.getvalue()) / 1024
        return f"{size_kb:.2f} KB" if size_kb < 1024 else f"{size_kb/1024:.2f} MB"

def safe_load_image(file) -> Image.Image:
    """
    Attempts to open and validate image file.

    Args:
        file (file-like): Uploaded file

    Returns:
        PIL.Image or None
    """
    try:
        image = Image.open(file)
        img_format = image.format
        if not validate_image_format(img_format):
            st.warning(f"Unsupported format: {img_format}")
            return None
        image = image.convert("RGB")
        return image
    except Exception as e:
        st.error(f"Corrupt or unreadable image: {file.name}")
        return None

