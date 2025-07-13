import os

class Config:
    # API Configuration
    API_URL = os.getenv("API_URL", "http://localhost:8000/api/predict")
    TIMEOUT = 30.0
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    # UI Configuration
    ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"]
    DEFAULT_IMAGE_SIZE = (224, 224)
    
    # Cache Configuration
    CACHE_TTL = 3600  # 1 hour

config = Config()