import streamlit as st
import requests
import io
import base64
import json
from PIL import Image
from typing import List, TypedDict, Optional
from threading import Thread
import time

# ====================== CONSTANTS & CONFIG ======================
API_URL = "http://localhost:8000/predict"
SUPPORTED_FORMATS = ["png", "jpg", "jpeg", "bmp"]
MODEL_OPTIONS = ["ResNet50", "ViT", "MobileNet"]
MAX_DISPLAY_DIM = 800  # For UI previews
MAX_API_DIM = 512      # For API processing
DEFAULT_COMPRESSION = 85