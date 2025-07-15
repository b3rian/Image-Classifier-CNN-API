from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import numpy as np
from PIL import Image
import io
import time
import tensorflow as tf
import json
import uvicorn

# Load ImageNet class names
with open("imagenet_class_index.json", "r") as f:
    class_idx = json.load(f)
    CLASS_NAMES = [class_idx[str(k)][1].replace("_", " ") for k in range(1000)]

# Load Keras model
MODEL_PATH = "D:\Telegram Desktop\custom_cnn_model_1000_classes.keras" 
model = tf.keras.models.load_model(MODEL_PATH)

# FastAPI app
app = FastAPI(
    title="Image Classifier API",
    description="FastAPI backend for AI Image Classifier",
    version="1.0"
)

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prediction response format
class Prediction(BaseModel):
    label: str
    confidence: float

class ApiResponse(BaseModel):
    predictions: List[Prediction]
    model_version: str
    inference_time: float

# Preprocess function
def preprocess_image(image_bytes: bytes, target_size=(480, 480)) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image).astype("float32") / 255.0
    return np.expand_dims(image_array, axis=0)

# Inference route
@app.post("/predict", response_model=ApiResponse)
async def predict(
    file: UploadFile = File(...),
    model: str = Query(default="Custom ResNet", description="Model name from frontend")
):
    try:
        contents = await file.read()
        input_tensor = preprocess_image(contents)

        start = time.time()
        predictions = model.predict(input_tensor)[0]
        end = time.time()

        top_indices = predictions.argsort()[-5:][::-1]
        results = [
            {"label": CLASS_NAMES[i], "confidence": float(predictions[i] * 100)}
            for i in top_indices
        ]

        return JSONResponse(
            status_code=200,
            content={
                "predictions": results,
                "model_version": model,
                "inference_time": round(end - start, 4)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
 
# =================== Health Check ===================
@app.get("/")
def root():
    return {"message": "Image Classifier API is running."}

# =================== Run Server ===================
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
