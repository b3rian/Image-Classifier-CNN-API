import os
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.sessions import SessionMiddleware
from app.config import settings  # Config module
from app.endpoints import router as prediction_router
from app.middleware import LoggingMiddleware  # Custom middleware
import logging
from prometheus_fastapi_instrumentator import Instrumentator  # For metrics

def create_app() -> FastAPI:
    """Factory function to create and configure the FastAPI application."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    app = FastAPI(
        title="Convolutional Neural Network(CNN) based Image Classification API",
        description=(
            "A FastAPI service for general image classification "
            "using custom_cnn_model_1000_classes model"
        ),
        version=1.0.0,
        docs_url="/docs"  # Use default docs URL
        redoc_url="/redoc",  # Typically just use one docs UI
        openapi_url="/openapi.json" if settings.DOCS_ENABLED else None
    )

    # Register routes
    app.include_router(
        prediction_router,
        prefix="/api",  # From config
        tags=["Prediction"]
    )
