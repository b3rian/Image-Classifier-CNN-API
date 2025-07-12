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

# Configure logging
logger = logging.getLogger(__name__)

# Metrics
PREDICTION_COUNTER = Counter('model_predictions_total', 'Total prediction requests')
PREDICTION_ERRORS = Counter('model_prediction_errors_total', 'Total prediction failures')
PREDICTION_LATENCY = Histogram('model_prediction_latency_seconds', 'Prediction latency')

# Configuration
MODEL_PATH = os.getenv('MODEL_PATH', "D:\Telegram Desktop\custom_cnn_model_1000_classes.keras")
CLASS_LABELS = [...]  # Actual class labels list/dict
MAX_RETRIES = 3

# Thread-safe model
_model: Optional[tf.keras.Model] = None
_model_lock = threading.Lock()
_model_loaded = False

def load_model() -> tf.keras.Model:
    """Thread-safe model loader with retries and health checks."""
    global _model, _model_loaded
    
    with _model_lock:
        if _model is None and not _model_loaded:
            try:
                logger.info("Loading model from %s", MODEL_PATH)
                
                # Verify model exists before loading
                if not os.path.exists(MODEL_PATH):
                    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
                
                _model = tf.keras.models.load_model(MODEL_PATH)
                
                # Verify model is usable
                test_input = np.zeros((1, 480, 480, 3), dtype=np.float32)
                _model.predict(test_input, verbose=0)
                
                _model_loaded = True
                logger.info("Model loaded and verified successfully")
                
            except Exception as e:
                _model_loaded = False
                logger.error("Model loading failed: %s", str(e))
                raise RuntimeError("Model initialization failed") from e
                
    if _model is None:
        raise RuntimeError("Model not available")
        
    return _model

@PREDICTION_LATENCY.time()
def predict(image_pil: PILImage, top_k: int = 3) -> List[Dict[str, float]]:
    """
    Production-grade prediction with error handling and metrics.
    
    Args:
        image_pil: Input image in PIL format
        top_k: Number of top predictions to return
        
    Returns:
        List of prediction dicts with keys 'label' and 'confidence'
        
    Raises:
        RuntimeError: For prediction failures
        ValueError: For invalid inputs
    """
    PREDICTION_COUNTER.inc()
    
    try:
        # Input validation
        if not isinstance(image_pil, PILImage):
            raise ValueError("Invalid image type")
            
        if top_k <= 0:
            raise ValueError("top_k must be positive")
            
        # Load model with retries
        model = None
        for attempt in range(MAX_RETRIES):
            try:
                model = load_model()
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    PREDICTION_ERRORS.inc()
                    logger.error("Model loading failed after %d attempts", MAX_RETRIES)
                    raise
                logger.warning("Model loading attempt %d failed, retrying...", attempt + 1)
                time.sleep(1)
                