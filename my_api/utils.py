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

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =================== Response Schemas ===================
class Prediction(BaseModel):
    label: str
    confidence: float

class ApiResponse(BaseModel):
    predictions: List[Prediction]
    model_version: str
    inference_time: float

# =================== Image Preprocessing ===================
def preprocess_image(
    image_bytes: bytes,
    target_size: tuple,
    preprocess_func: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image).astype("float32")
    image_array = preprocess_func(image_array)
    return np.expand_dims(image_array, axis=0)

# =================== Inference Endpoint ===================
@app.post("/predict", response_model=ApiResponse)
async def predict(
    file: UploadFile = File(...),
    model_name: str = Query(..., description="Choose 'efficientnet' or 'resnet'")
):
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not found. Available options: {list(MODEL_REGISTRY.keys())}"
        )

    try:
        model = models[model_name]
        config = MODEL_REGISTRY[model_name]
        contents = await file.read()
        
        # Preprocess
        input_tensor = preprocess_image(contents, config["input_size"], config["preprocess"])

        # Inference
        start = time.time()
        predictions = model.predict(input_tensor)
        end = time.time()

        # Decode predictions
        decoded = config["decode"](predictions, top=3)[0]
        results = [
            {"label": label.replace("_", " "), "confidence": float(score * 100)}
            for (_, label, score) in decoded
        ]

        return JSONResponse(
            status_code=200,
            content={
                "predictions": results,
                "model_version": model_name,
                "inference_time": round(end - start, 4)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")