from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from my_APIs.endpoints import router as prediction_router

def create_app() -> FastAPI:
    """
    Factory function to create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI app instance.
    """
    app = FastAPI(
        title="EfficientNetV2L Image Classification API",
        description=(
            "A production-grade FastAPI service that accepts image uploads "
            "and returns the top-3 predicted classes using an EfficientNetV2L model."
        ),
        version="1.0.0",
        docs_url="/docs",  # Swagger UI
        redoc_url="/redoc"  # ReDoc UI
    )

    # Register API routes
    app.include_router(
        prediction_router,
        prefix="/api",
        tags=["Prediction"]
    )

    # Middleware Configuration (e.g., for CORS)
    configure_middlewares(app)

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
