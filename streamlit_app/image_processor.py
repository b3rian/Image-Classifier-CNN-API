from PIL import Image
import io
import numpy as np
import streamlit as st
from config import config

class ImageProcessor:
    @staticmethod
    def validate_image(file) -> bool:
        """Check if file meets requirements"""
        if file.size > config.MAX_FILE_SIZE:
            st.error(f"File too large (max {config.MAX_FILE_SIZE//1024//1024}MB)")
            return False
        return True
