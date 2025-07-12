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