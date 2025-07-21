"""Service functions for handling model predictions and image preprocessing."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from typing import Callable
import io
from api_backend.configs import logger
from api_backend.models import InvalidImageError

executor = ThreadPoolExecutor(max_workers=4) # Adjust based on expected concurrency

async def async_predict(model, input_tensor):
    """Run model prediction in a separate thread."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, model.predict, input_tensor)

def preprocess_image(
    image_bytes: bytes,
    target_size: tuple,
    preprocess_func: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Preprocess image bytes into model input tensor."""
    try:
        # Load image from bytes and convert to RGB
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(target_size)
        image_array = np.array(image).astype("float32")
        image_array = preprocess_func(image_array)
        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        raise InvalidImageError(f"Invalid image file: {str(e)}")