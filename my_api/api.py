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
import json
import urllib.request

# Download the ImageNet labels file
url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
with urllib.request.urlopen(url) as response:
    imagenet_class_index = json.load(response)

# =================== Config ===================
MODEL_PATH = "cnn_model.keras"
CLASS_NAMES = [imagenet_class_index[str(k)][1].replace('_', ' ') for k in range(1000)]  # Class labels
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

# =================== API Endpoints ===================
@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...), model: str = "Custom ResNet"):
    start_time = time.time()

    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        raise HTTPException(status_code=400, detail="Unsupported image format.")

    contents = await file.read()
    image_tensor = read_image_as_tensor(contents)
    predictions = predict_image(image_tensor)

    response = {
        "predictions": predictions,
        "model_version": MODEL_PATH,
        "inference_time": round((time.time() - start_time) * 1000, 2)  # ms
    }

    return JSONResponse(content=response)

# =================== Health Check ===================
@app.get("/")
def root():
    return {"message": "Image Classifier API is running."}

# =================== Run Server ===================
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
