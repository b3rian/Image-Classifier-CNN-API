"FastAPI backend for AI Image Classifier with multiple Keras models"

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware import Middleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import List, Callable, Optional
from enum import Enum
import numpy as np
import os
from PIL import Image
import io
import time
import tensorflow as tf
import uvicorn
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import datetime

# =================== Configurations ===================
class Settings(BaseSettings):
    """Application configuration settings."""
    models_dir: str = "models"
    allowed_origins: list[str] = ["*"]
    app_name: str = "Image Classifier API"
    app_version: str = "1.1.0"
    log_level: str = "INFO"
    enable_https_redirect: bool = False
    
    class Config:
        env_file = ".env"

settings = Settings()

# =================== Logging Setup ===================
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =================== Model paths from Docker container ===================
MODEL_DIR = os.getenv("MODEL_DIR", "models")
resnet_model = os.path.join(MODEL_DIR, "resnet50_imagenet.keras")
efficientnet_model = os.path.join(MODEL_DIR, "efficientnet.keras")

# =================== Model Registry ===================
MODEL_REGISTRY = {
    "efficientnet": {
        "path": efficientnet_model,
        "preprocess": tf.keras.applications.efficientnet_v2.preprocess_input,
        "decode": tf.keras.applications.efficientnet_v2.decode_predictions,
        "input_size": (480, 480)
    },
    "resnet": {
        "path": resnet_model,
        "preprocess": tf.keras.applications.resnet50.preprocess_input,
        "decode": tf.keras.applications.resnet50.decode_predictions,
        "input_size": (224, 224)
    }
}

# =================== Custom Exceptions ===================
class ModelNotFoundError(Exception):
    """Exception raised when a requested model is not found."""
    pass

class InvalidImageError(Exception):
    """Exception raised for invalid image files."""
    pass

# =================== Model Loading ===================
@lru_cache(maxsize=None) # Cache loaded models to avoid reloading
def load_model(model_path: str, input_size: tuple) -> tf.keras.Model:
    """Load a Keras model and perform a warm-up inference to ensure it's ready for predictions."""
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

# Initialize models with error handling
models = {}
for name, config in MODEL_REGISTRY.items():
    try:
        models[name] = load_model(config["path"], config["input_size"])
    except Exception as e:
        logger.error(f"Could not load model {name}: {str(e)}")

# =================== FastAPI Application Setup ===================
middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

if settings.enable_https_redirect:
    # Add HTTPS redirect middleware if enabled in settings
    middleware.append(Middleware(HTTPSRedirectMiddleware))

# Create FastAPI app with middleware and settings
app = FastAPI(
    title=settings.app_name,
    description="FastAPI backend for AI Image Classifier with multiple Keras models",
    version=settings.app_version,
    contact={
        "name": "Backend Team",
        "email": "brayan.8189@gmail.com",
    },
    license_info={
        "name": "MIT",
    },
    openapi_tags=[{
        "name": "predictions",
        "description": "Operations with image predictions",
    }],
    middleware=middleware
)

# =================== Request Logging Middleware ===================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log incoming requests and their processing time."""
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    logger.info(
        f"Request: {request.method} {request.url} completed in {process_time:.2f}ms"
    )
    return response

# =================== Error Handlers ===================
@app.exception_handler(ModelNotFoundError)
async def model_not_found_handler(request, exc):
    """Handle ModelNotFoundError exceptions."""
    return JSONResponse(
        status_code=404,
        content={"message": str(exc)},
    )

@app.exception_handler(InvalidImageError)
async def invalid_image_handler(request, exc):
    """Handle InvalidImageError exceptions."""
    return JSONResponse(
        status_code=400,
        content={"message": str(exc)},
    )

# =================== Response Schemas ===================
class Prediction(BaseModel):
    """Schema for individual prediction results."""
    label: str
    confidence: float = Field(..., ge=0.0, le=100.0)  

class ApiResponse(BaseModel):
    """Schema for API response containing predictions."""
    predictions: List[Prediction]
    model_version: str
    inference_time: float
    timestamp: str

class HealthCheckResponse(BaseModel):
    """Schema for health check response."""
    status: str
    models_loaded: List[str]
    timestamp: str

# =================== Model Name Enum ===================
class ModelName(str, Enum):
    """Enum for available model names."""
    efficientnet = "efficientnet"
    resnet = "resnet"

# =================== Image Preprocessing ===================
def preprocess_image(
    image_bytes: bytes,
    target_size: tuple,
    preprocess_func: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Preprocess image bytes into a tensor suitable for model input."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(target_size)
        image_array = np.array(image).astype("float32")
        image_array = preprocess_func(image_array)
        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        raise InvalidImageError(f"Invalid image file: {str(e)}")

# =================== Async Prediction ===================
executor = ThreadPoolExecutor(max_workers=4)

async def async_predict(model: tf.keras.Model, input_tensor: np.ndarray):
    """Run model prediction in an asynchronous manner using a thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, model.predict, input_tensor)

# =================== Inference Endpoint ===================
@app.post("/predict", response_model=ApiResponse, tags=["predictions"])
async def predict(
    request: Request,
    file: UploadFile = File(...),
    model_name: ModelName = Query(..., description="Choose model for inference")
):
    """Endpoint to classify an image using the specified model."""
    if model_name.value not in models:
        logger.error(f"Model '{model_name}' not found in loaded models")
        raise ModelNotFoundError(
            f"Model '{model_name}' not available. Available options: {list(models.keys())}"
        )

    try:
        # Load model and configuration
        model = models[model_name.value]
        config = MODEL_REGISTRY[model_name.value]
        contents = await file.read()
        
        # Preprocess
        input_tensor = preprocess_image(contents, config["input_size"], config["preprocess"])

        # Inference
        start = time.time()
        predictions = await async_predict(model, input_tensor)
        end = time.time()

        # Decode predictions
        decoded = config["decode"](predictions, top=3)[0]
        results = [
            {"label": label.replace("_", " "), "confidence": round(float(score * 100), 2)}
            for (_, label, score) in decoded
        ]

        return {
            "predictions": results,
            "model_version": model_name.value,
            "inference_time": round(end - start, 4),
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
    except InvalidImageError as e:
        raise
    except Exception as e:
        logger.error(f"Inference error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    
# =================== Health Check Endpoints ===================
@app.get("/", include_in_schema=False)
def root():
    """Root endpoint to check if the API is running."""
    return {"message": "Image Classifier API is running."}

@app.get("/health", response_model=HealthCheckResponse, tags=["health"])
async def health_check():
    """Health check endpoint to verify API status and loaded models."""
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

# =================== Run Server ===================
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower()
    )