from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Callable
import numpy as np
from PIL import Image
import io
import time
import tensorflow as tf
import uvicorn

# Keras preprocessors and decoders
from tensorflow.keras.applications.efficientnet_v2 import (
    preprocess_input as efficientnet_preprocess,
    decode_predictions as efficientnet_decode
)
from tensorflow.keras.applications.resnet50 import (
    preprocess_input as resnet_preprocess,
    decode_predictions as resnet_decode
)

# =================== Model Registry ===================
MODEL_REGISTRY = {
    "efficientnet": {
        "path": "D:/Telegram Desktop/efficientnet_model.keras",
        "preprocess": efficientnet_preprocess,
        "decode": efficientnet_decode,
        "input_size": (480, 480)
    },
    "resnet": {
        "path": "D:/Telegram Desktop/custom_cnn_model_1000_classes.keras",
        "preprocess": resnet_preprocess,
        "decode": resnet_decode,
        "input_size": (224, 224)
    }
}

# Load models into memory
models = {
    name: tf.keras.models.load_model(config["path"])
    for name, config in MODEL_REGISTRY.items()
}

# =================== FastAPI Setup ===================
app = FastAPI(
    title="Image Classifier API",
    description="FastAPI backend for AI Image Classifier with multiple Keras models",
    version="1.1"
)