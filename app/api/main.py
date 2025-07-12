import os
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.sessions import SessionMiddleware
from api.endpoints import router as prediction_router
import logging
from prometheus_fastapi_instrumentator import Instrumentator  # For metrics

def create_app() -> FastAPI:
    """
    Factory function to create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI app instance.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    app = FastAPI(
        title="Convolutional Neural Network(CNN) based Image Classification API",
        description=(
            "A FastAPI service that accepts image uploads "
            "and returns the top-3 predicted classes using a custom CNN model with 1000 classes."
        ),
        version="1.0.0",
        docs_url="/docs",  # Swagger UI
        redoc_url="/redoc"  # ReDoc UI
    )

    # Register routes
    app.include_router(
        prediction_router,
        prefix="/api",  # From config
        tags=["Prediction"]
    )

    # Configure middlewares
    configure_middlewares(app)
    
    # Add startup/shutdown events
    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting application...")
        # Initialize connections, warm up models, etc.
        Instrumentator().instrument(app).expose(app)  # Metrics endpoint
        
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down application...")
        # Clean up resources
        
    return app

def configure_middlewares(app: FastAPI) -> None:
    """
    Configures global middlewares for the FastAPI application.  
 
    Args:
        app (FastAPI): The FastAPI app instance.
    """
    # Allow all origins during development; restrict in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Use specific domains in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Instantiate the app
app = create_app()
