"""
predictor.py

Handles model loading and image prediction logic using EfficientNetV2L.
This module ensures efficient, thread-safe inference optimized for limited compute environments,
such as Hugging Face Spaces (2 vCPU, 16GB RAM).
"""
import tensorflow as tf
import numpy as np
import logging
from typing import List, Dict, Optional
from PIL.Image import Image as PILImage
from app.utils import preprocess_image
import threading
import os
from prometheus_client import Counter, Histogram