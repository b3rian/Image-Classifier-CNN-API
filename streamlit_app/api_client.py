import httpx
import logging
from streamlit_app.config import config

logger = logging.getLogger(__name__)

class APIClient:
    @staticmethod
    async def classify_image(image_bytes: bytes) -> dict:
        """
        Send image to classification API
        Args:
            image_bytes: Binary image data
        Returns:
            dict: API response with predictions
        """
        try:
            async with httpx.AsyncClient(timeout=config.TIMEOUT) as client:
                files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
                response = await client.post(config.API_URL, files=files)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"API Error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise

    @staticmethod
    def get_class_labels() -> list:
        """Get available class labels (cached)"""
        try:
            response = httpx.get(f"{config.API_URL.rsplit('/', 1)[0]}/classes")
            return response.json()
        except Exception as e:
            logger.warning(f"Couldn't fetch class labels: {str(e)}")
            return []