from functools import lru_cache
import pandas as pd
from config import config

class CacheManager:
    @staticmethod
    @lru_cache(maxsize=128)
    def get_cached_labels():
        """Cache class labels"""
        from streamlit_app.api_client import APIClient
        return APIClient.get_class_labels()