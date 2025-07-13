"""Production-ready EfficientNetV2L predictor with dynamic label loading."""

import tensorflow as tf
import numpy as np
import logging
from typing import List, Dict, Optional
from PIL.Image import Image as PILImage
from my_APIs.utils import preprocess_image
import threading
import io
import os
import json
import requests
from prometheus_client import Counter, Histogram
import time

# Configure logging
logger = logging.getLogger(__name__)

# Metrics
PREDICTION_COUNTER = Counter('model_predictions_total', 'Total prediction requests')
PREDICTION_ERRORS = Counter('model_prediction_errors_total', 'Total prediction failures')
PREDICTION_LATENCY = Histogram('model_prediction_latency_seconds', 'Prediction latency')

# Configuration
MODEL_PATH = os.getenv('MODEL_PATH', "D:/Telegram Desktop/custom_cnn_model_1000_classes.keras" )
LABELS_URL = os.getenv('LABELS_URL', 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json')
MAX_RETRIES = 3

# Thread-safe model and labels
_model: Optional[tf.keras.Model] = None
_model_lock = threading.Lock()
_class_labels: Optional[Dict[int, str]] = None
_labels_lock = threading.Lock()

def load_labels() -> Dict[int, str]:
    """
    Load class labels either from URL or local cache with thread safety.
    
    Returns:
        Dictionary mapping class indices to human-readable labels
    """
    global _class_labels
    
    with _labels_lock:
        if _class_labels is None:
            try:
                logger.info("Loading class labels from %s", LABELS_URL)
                
                # Try remote URL first
                response = requests.get(LABELS_URL, timeout=10)
                response.raise_for_status()
                label_map = response.json()
                
                # Convert to {int: str} format
                _class_labels = {
                    int(class_id): class_name[1]  # Using human-readable name
                    for class_id, class_name in label_map.items()
                }
                
                logger.info("Loaded %d class labels", len(_class_labels))
                
            except Exception as e:
                logger.error("Failed to load remote labels: %s", str(e))
                
                # Fallback to local labels if available
                local_path = os.path.join(os.path.dirname(MODEL_PATH), "labels.json")
                if os.path.exists(local_path):
                    logger.info("Falling back to local labels at %s", local_path)
                    with open(local_path) as f:
                        _class_labels = json.load(f)
                else:
                    raise RuntimeError("No labels available - both remote and local failed")
    
    return _class_labels

def load_model() -> tf.keras.Model:
    """Thread-safe model loader with label pre-loading."""
    global _model
    
    with _model_lock:
        if _model is None:
            try:
                logger.info("Loading model from %s", MODEL_PATH)
                
                # Pre-load labels first
                load_labels()
                
                # Load model
                _model = tf.keras.models.load_model(MODEL_PATH)
                
                # Verify model and labels compatibility
                if len(_class_labels) != _model.output_shape[1]:
                    logger.warning(
                        "Label count (%d) doesn't match model outputs (%d)",
                        len(_class_labels), _model.output_shape[1]
                    )
                
                logger.info("Model loaded successfully with %d classes", len(_class_labels))
                
            except Exception as e:
                _model = None
                logger.error("Model loading failed: %s", str(e))
                raise RuntimeError("Model initialization failed") from e
    
    return _model

@PREDICTION_LATENCY.time()
def predict(image_pil: PILImage, top_k: int = 3) -> List[Dict[str, float]]:
    """
    Production-grade prediction with dynamic label mapping.
    
    Args:
        image_pil: Input image in PIL format (RGB)
        top_k: Number of top predictions to return
        
    Returns:
        List of predictions with 'label', 'confidence', and 'class_id'
        
    Raises:
        RuntimeError: For prediction failures
        ValueError: For invalid inputs
    """
    PREDICTION_COUNTER.inc()
    start_time = time.time()
    
    try:
        # Input validation
        if not isinstance(image_pil, PILImage):
            raise ValueError("Input must be a PIL Image")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        
        # Load resources with retries
        model = None
        for attempt in range(MAX_RETRIES):
            try:
                model = load_model()
                labels = load_labels()
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    PREDICTION_ERRORS.inc()
                    logger.error("Resource loading failed after %d attempts", MAX_RETRIES)
                    raise RuntimeError("Service unavailable") from e
                time.sleep(1)
        
        # Preprocess
        try:
            image_array = preprocess_image(image_pil, target_size=(480, 480))
            input_tensor = np.expand_dims(image_array, axis=0)
        except Exception as e:
            raise ValueError("Image preprocessing failed") from e
        
        # Inference
        try:
            predictions = model.predict(input_tensor, verbose=0)[0]
        except Exception as e:
            PREDICTION_ERRORS.inc()
            logger.error("Prediction failed: %s", str(e))
            raise RuntimeError("Prediction service error") from e
        
        # Process results
        top_indices = predictions.argsort()[-top_k:][::-1]
        top_scores = predictions[top_indices]
        
        return [
            {
                "label": labels.get(int(idx), f"class_{idx}"),
                "confidence": float(score),
                "class_id": int(idx),
                "model": "EfficientNetV2L",
                "timestamp": time.time(),
                "processing_ms": (time.time() - start_time) * 1000
            }
            for idx, score in zip(top_indices, top_scores)
        ]
        
    except Exception as e:
        PREDICTION_ERRORS.inc()
        logger.error("Prediction error: %s", str(e), exc_info=True)
        raise

def get_available_labels() -> Dict[int, str]:
    """API-accessible method to get all loaded labels"""
    return load_labels()