import logging
from api_backend.configs import settings

def configure_logging():
    """Configure application logging."""
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)

logger = configure_logging()