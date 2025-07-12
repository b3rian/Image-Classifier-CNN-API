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
 

 