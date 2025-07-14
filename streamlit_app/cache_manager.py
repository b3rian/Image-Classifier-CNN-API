from functools import lru_cache
import pandas as pd
from streamlit_app.config import config
import streamlit as st

class CacheManager:
    @staticmethod
    @lru_cache(maxsize=128)
    def get_cached_labels():
        """Cache class labels"""
        from streamlit_app.api_client import APIClient
        return APIClient.get_class_labels()
    
    @staticmethod
    def cache_prediction(image_hash: str, predictions: dict):
        """Store prediction results"""
        if not hasattr(st, 'session_state'):
            return
        
        if 'predictions_cache' not in st.session_state:
            st.session_state.predictions_cache = pd.DataFrame(
                columns=['image_hash', 'predictions', 'timestamp']
            )
        
        st.session_state.predictions_cache.loc[len(st.session_state.predictions_cache)] = {
            'image_hash': image_hash,
            'predictions': predictions,
            'timestamp': pd.Timestamp.now()
        }