from functools import lru_cache
import numpy as np
import tensorflow as tf
from api_backend.configs import logger
import os

# Get the base directory of the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Model Registry
MODEL_REGISTRY = {
    "efficientnet": {
        "path": "D:/Documents/models/efficientnet.keras",
        "preprocess": tf.keras.applications.efficientnet_v2.preprocess_input,
        "decode": tf.keras.applications.efficientnet_v2.decode_predictions,
        "input_size": (480, 480)
    },
    "resnet": {
        "path": "D:/Documents/models/resnet50_imagenet.keras",
        "preprocess": tf.keras.applications.resnet50.preprocess_input,
        "decode": tf.keras.applications.resnet50.decode_predictions,
        "input_size": (224, 224)
    }
}

# Exceptions
class ModelNotFoundError(Exception):
    """Exception raised when a requested model is not found."""
    pass

class InvalidImageError(Exception):
    """Exception raised when image processing fails."""
    pass

# Model Loading
@lru_cache(maxsize=None)
def load_model(model_path: str, input_size: tuple) -> tf.keras.Model:
    """Load and warm up a TensorFlow model with caching."""
    try:
        model = tf.keras.models.load_model(model_path)
        # Warm up the model
        dummy_input = np.zeros((1, *input_size, 3))
        _ = model.predict(dummy_input)
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {str(e)}")
        raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")

# Initialize models
models = {}
for name, config in MODEL_REGISTRY.items():
    try:
        models[name] = load_model(config["path"], config["input_size"])
    except Exception as e:
        logger.error(f"Could not load model {name}: {str(e)}")