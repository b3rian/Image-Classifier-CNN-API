from PIL import Image
import io
import numpy as np
import streamlit as st
from streamlit_app.config import config

class ImageProcessor:
    @staticmethod
    def validate_image(file) -> bool:
        """Check if file meets requirements"""
        if file.size > config.MAX_FILE_SIZE:
            st.error(f"File too large (max {config.MAX_FILE_SIZE//1024//1024}MB)")
            return False
        return True

    @staticmethod
    def preprocess_image(file) -> bytes:
        """Convert image to optimized format"""
        img = Image.open(io.BytesIO(file.getvalue()))
        img = img.resize(config.DEFAULT_IMAGE_SIZE)
        
        # Convert to bytes with optimized quality
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=85)
        return img_byte_arr.getvalue()
    
    @staticmethod
    def display_with_overlay(image_bytes, predictions):
        """Generate image with annotation overlay"""
        img = Image.open(io.BytesIO(image_bytes))
        # visualization logic here
        return img