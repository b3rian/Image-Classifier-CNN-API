import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from PIL import Image, UnidentifiedImageError
import numpy as np
import io
from typing import List, Dict
from pydantic import BaseModel
from datetime import datetime
import os
from app.predictor import predict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_CONTENT_TYPES = ["image/jpeg", "image/png", "image/jpg"]

# Response model
class PredictionResult(BaseModel):
    class_label: str
    confidence: float

class PredictionResponse(BaseModel):
    predictions: List[PredictionResult]
    processing_time: float
    timestamp: str

@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"description": "Invalid input file"},
        413: {"description": "File too large"},
        500: {"description": "Internal processing error"},
    },
    summary="Classify an image",
    description="Accepts an image file and returns top predictions with confidence scores.",
)

async def predict_image(
    request: Request,
    file: UploadFile = File(...),
):
    """
    Endpoint to receive image file, preprocess, and return top predictions.
    
    Args:
        file: Uploaded image file (JPG/PNG)
        
    Returns:
        JSON with top predictions and metadata
    """
    start_time = datetime.now()
    
    try:
        # 1. Validate file size
        file_size = request.headers.get("content-length", 0)
        if int(file_size) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Max size: {MAX_FILE_SIZE/1024/1024}MB",
            )

        # 2. Validate content type
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}",
            )

        # 3. Read and validate image
        image_bytes = await file.read()
        
        # Additional check for actual file size
        if len(image_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Max size: {MAX_FILE_SIZE/1024/1024}MB",
            )

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image.verify()  # Verify it's actually an image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Need to reopen after verify
        except (UnidentifiedImageError, IOError) as e:
            logger.error(f"Invalid image file: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image file",
            )


        # 4. Run prediction
        logger.info(f"Processing image: {file.filename}")
        top_preds = predict(image)

        # 5. Prepare response
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response_data = {
            "predictions": top_preds,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
        }
        
        logger.info(
            f"Prediction completed for {file.filename}. "
            f"Processing time: {processing_time:.2f}s"
        )
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise  # Re-raise FastAPI HTTP exceptions
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing image",
        )