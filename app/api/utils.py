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
from typing import Tuple, Optional, Union
import logging
from dataclasses import dataclass
import time
from functools import wraps

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