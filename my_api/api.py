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
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input, decode_predictions

# Load Keras model
MODEL_PATH = "D:/Telegram Desktop/custom_cnn_model_1000_classes.keras" 
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
    image_array = np.array(image).astype("float32")
    image_array = preprocess_input(image_array)  # <-- This is crucial
    return np.expand_dims(image_array, axis=0)


# Inference route
@app.post("/predict", response_model=ApiResponse)
async def predict(
    file: UploadFile = File(...),
    model_name: str = Query(default="Custom ResNet", description="Model name from frontend")
):
    try:
        contents = await file.read()
        input_tensor = preprocess_image(contents)

        start = time.time()
        predictions = model.predict(input_tensor)
        end = time.time()

        decoded = decode_predictions(predictions, top=3)[0]
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
 
# =================== Health Check ===================
@app.get("/")
def root():
    return {"message": "Image Classifier API is running."}

# =================== Run Server ===================
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
