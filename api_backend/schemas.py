from pydantic import BaseModel, Field
from enum import Enum
from typing import List
import datetime

# Response Models
class Prediction(BaseModel):
    """Single prediction result schema."""
    label: str
    confidence: float = Field(..., ge=0.0, le=100.0)  

class ApiResponse(BaseModel):
    """API response schema for prediction endpoint."""
    predictions: List[Prediction]
    model_version: str
    inference_time: float
    timestamp: str

class HealthCheckResponse(BaseModel):
    """Health check response schema."""
    status: str
    models_loaded: List[str]
    timestamp: str

# Enums
class ModelName(str, Enum):
    """Supported model names enumeration."""
    efficientnet = "efficientnet"
    resnet = "resnet"