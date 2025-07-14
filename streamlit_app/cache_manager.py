import streamlit as st
import hashlib
import pandas as pd
from datetime import datetime

class CacheManager:
    @staticmethod
    def _generate_cache_key(file_bytes: bytes) -> str:
        """Generate MD5 hash key for cached images"""
        return hashlib.md5(file_bytes).hexdigest()

    @staticmethod
    def check_cache(file_hash: str) -> dict:
        """
        Check if results exist in cache
        Returns:
            dict: Cached predictions if exists, else None
        """
        if 'prediction_cache' not in st.session_state:
            st.session_state.prediction_cache = {}
        return st.session_state.prediction_cache.get(file_hash)

    @staticmethod
    def cache_prediction(file_hash: str, predictions: dict):
        """Store prediction results in cache"""
        if 'prediction_cache' not in st.session_state:
            st.session_state.prediction_cache = {}
        st.session_state.prediction_cache[file_hash] = predictions

    @staticmethod
    def clear_cache():
        """Clear all cached predictions"""
        if 'prediction_cache' in st.session_state:
            del st.session_state.prediction_cache
            st.success("Cache cleared successfully!")
    
     
        