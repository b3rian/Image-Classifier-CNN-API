"Configuration module for the Image Classifier API backend."

from pydantic_settings import BaseSettings
import logging

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

# Initialize settings and logging
settings = Settings()
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)