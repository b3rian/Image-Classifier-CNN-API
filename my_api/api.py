from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import time
import uvicorn

# =================== Config ===================
MODEL_PATH = "cnn_model.keras"
CLASS_NAMES = ["cat", "dog", "bird", "other"]  # Class labels
IMAGE_SIZE = (480, 480)  # Match model input size

# =================== Load Model ===================
print("🔄 Loading model...")
model = load_model(MODEL_PATH)
inference_model = tf.function(model)  # Optimized inference
print("✅ Model loaded successfully.")

# =================== FastAPI App ===================
app = FastAPI(
    title="Image Classifier API",
    description="FastAPI backend for image classification with CNN",
    version="1.0.0"
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =================== Helper Functions ===================
def read_image_as_tensor(file: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(file)).convert("RGB")
        image = image.resize(IMAGE_SIZE)
        image_array = np.asarray(image) / 255.0  # Normalize to [0, 1]
        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    
def predict_image(image_tensor: np.ndarray):
    preds = inference_model(image_tensor).numpy()[0]
    sorted_indices = preds.argsort()[::-1]
    results = [
        {
            "label": CLASS_NAMES[i],
            "confidence": round(float(preds[i]) * 100, 2)
        }
        for i in sorted_indices
    ]
    return results
