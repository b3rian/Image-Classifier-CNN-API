import pytest
from fastapi.testclient import TestClient
from fastapi import status, UploadFile
from io import BytesIO
from PIL import Image
import numpy as np
import os
from datetime import datetime
from unittest.mock import patch, MagicMock
from api.endpoints import router
from fastapi import FastAPI

# Create a FastAPI app and include the router
app = FastAPI()
app.include_router(router)

client = TestClient(app)

# Test data
TEST_IMAGE_PATH = "test_image.jpg"
TEST_LARGE_IMAGE_PATH = "large_image.jpg"

@pytest.fixture(scope="module", autouse=True)
def setup_test_images():
    """Create test images before tests and clean up after"""
    # Create a small test image
    img = Image.new('RGB', (100, 100), color='red')
    img.save(TEST_IMAGE_PATH)
    
    # Create a large test image (>10MB)
    large_img = Image.new('RGB', (2000, 2000))
    large_img.save(TEST_LARGE_IMAGE_PATH)
    
    yield  # Tests run here
    
    # Cleanup
    if os.path.exists(TEST_IMAGE_PATH):
        os.remove(TEST_IMAGE_PATH)
    if os.path.exists(TEST_LARGE_IMAGE_PATH):
        os.remove(TEST_LARGE_IMAGE_PATH)