from pydantic_settings import BaseSettings

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