"""
Enhanced image preprocessing utilities for EfficientNetV2L inference.

Features:
- Comprehensive input validation
- Detailed error handling
- Performance optimizations
- Support for batch processing
"""

import numpy as np
from PIL import Image, UnidentifiedImageError
from typing import Tuple, Optional, Union, List
import logging
from dataclasses import dataclass
import time
from functools import wraps
import io

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class PreprocessConfig:
    target_size: Tuple[int, int] = (480, 480)
    normalize: bool = True
    mean: Optional[Tuple[float, float, float]] = None
    std: Optional[Tuple[float, float, float]] = None

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        logger.debug(f"{func.__name__} executed in {elapsed:.4f} seconds")
        return result
    return wrapper

def validate_image_format(image: Image.Image) -> None:
    """Validates image meets processing requirements."""
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image, got {type(image)}")
    if image.mode != "RGB":
        raise ValueError(f"Image must be RGB. Got '{image.mode}'")
    if not image.size[0] or not image.size[1]:
        raise ValueError("Image has invalid dimensions")

@timing_decorator
def preprocess_image(
    image: Union[Image.Image, np.ndarray],
    config: PreprocessConfig = PreprocessConfig()
) -> np.ndarray:
    """
    Enhanced image preprocessing with:
    - Configurable normalization
    - Advanced validation
    - Performance tracking
    """
    try:
        # Convert input if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        
        # Validate and convert
        validate_image_format(image.convert("RGB"))  # Test conversion first
        image = image.convert("RGB")
        
        # Resize with anti-aliasing
        image_resized = image.resize(config.target_size, Image.Resampling.LANCZOS)
        
        # Convert to array
        image_array = np.asarray(image_resized, dtype=np.float32)
        
        # Advanced normalization
        if config.normalize:
            image_array /= 255.0
            if config.mean and config.std:
                image_array = (image_array - config.mean) / config.std
                
        return image_array
    
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise ValueError(f"Image preprocessing error: {str(e)}") from e

@timing_decorator
def load_image_from_bytes(
    image_bytes: bytes,
    validate: bool = True
) -> Image.Image:
    """Robust image loading with validation."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if validate:
            validate_image_format(image.convert("RGB"))
        return image
    except UnidentifiedImageError:
        logger.error("Failed to identify image from bytes")
        raise ValueError("Invalid image format")
    except Exception as e:
        logger.error(f"Image loading failed: {str(e)}")
        raise ValueError(f"Image loading error: {str(e)}") from e

def batch_preprocess(
    images: List[Union[Image.Image, np.ndarray]],
    config: PreprocessConfig = PreprocessConfig()
) -> np.ndarray:
    """Process multiple images efficiently."""
    return np.stack([preprocess_image(img, config) for img in images])