"Main entry point for the FastAPI application with multiple Keras models for image classification."

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request
from fastapi.middleware import Middleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
import datetime
import time
from api_backend.configs import settings, logger
from api_backend.models import models, MODEL_REGISTRY, ModelNotFoundError, InvalidImageError
from api_backend.schemas import ApiResponse, HealthCheckResponse, ModelName
from api_backend.services import async_predict, preprocess_image

# Setup middleware
middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

if settings.enable_https_redirect:
    "Enable HTTPS redirect middleware if configured"
    middleware.append(Middleware(HTTPSRedirectMiddleware))

# Create FastAPI app with configuration
app = FastAPI(
    title=settings.app_name,
    description="FastAPI backend for AI Image Classifier with multiple Keras models",
    version=settings.app_version,
    contact={
        "name": "Brian",
        "email": "brayann.8189@gmail.com",
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

# Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log request processing time."""
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    logger.info(
        f"Request: {request.method} {request.url} completed in {process_time:.2f}ms"
    )
    return response

# Exception Handlers
@app.exception_handler(ModelNotFoundError)
async def model_not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": str(exc)},
    )

@app.exception_handler(InvalidImageError)
async def invalid_image_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"message": str(exc)},
    )

# Endpoints
@app.post("/predict", response_model=ApiResponse, tags=["predictions"])
async def predict(
    request: Request,
    file: UploadFile = File(...),
    model_name: ModelName = Query(..., description="Choose model for inference")
):
    if model_name.value not in models:
        logger.error(f"Model '{model_name}' not found in loaded models")
        raise ModelNotFoundError(
            f"Model '{model_name}' not available. Available options: {list(models.keys())}"
        )

    try:
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
    
@app.get("/", include_in_schema=False)
def root():
    return {"message": "Image Classifier API is running."}

@app.get("/health", response_model=HealthCheckResponse, tags=["health"])
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "timestamp": datetime.datetime.utcnow().isoformat()
    }