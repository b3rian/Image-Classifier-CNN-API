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

def create_test_image_file(filename=TEST_IMAGE_PATH):
    """Helper to create an in-memory test image file"""
    with open(filename, "rb") as f:
        image_bytes = f.read()
    return ("file", image_bytes)

def test_predict_endpoint_success():
    """Test successful image prediction"""
    # Mock the prediction function to return consistent results
    mock_predictions = [
        {"class_label": "cat", "confidence": 0.95},
        {"class_label": "dog", "confidence": 0.05}
    ]
    
    with patch("api.endpoints.predict", return_value=mock_predictions) as mock_predict:
        files = {"file": open(TEST_IMAGE_PATH, "rb")}
        response = client.post("/predict", files=files)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check response structure
        assert "predictions" in data
        assert "processing_time" in data
        assert "timestamp" in data
        
        # Check predictions
        assert len(data["predictions"]) == 2
        assert data["predictions"][0]["class_label"] == "cat"
        assert data["predictions"][0]["confidence"] == 0.95
        
        # Check processing time is positive
        assert float(data["processing_time"]) > 0
        
        # Verify mock was called
        mock_predict.assert_called_once()