import pytest
from fastapi.testclient import TestClient
from fastapi import status, UploadFile
from io import BytesIO
from PIL import Image
import numpy as np
import os
from datetime import datetime
from unittest.mock import patch, MagicMock
import logging

# Import your router directly without relative import
from app.api.endpoints import router
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

def test_predict_endpoint_invalid_file_type():
    """Test with invalid file type"""
    # Create a test text file
    test_file = BytesIO(b"This is not an image")
    files = {"file": ("test.txt", test_file, "text/plain")}
    
    response = client.post("/predict", files=files)
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Invalid file type" in response.json()["detail"]

def test_predict_endpoint_file_too_large():
    """Test with file exceeding size limit"""
    files = {"file": open(TEST_LARGE_IMAGE_PATH, "rb")}
    
    response = client.post("/predict", files=files)
    
    assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    assert "File too large" in response.json()["detail"]

def test_predict_endpoint_corrupted_image():
    """Test with corrupted image file"""
    # Create a file that looks like an image but is corrupted
    corrupted_image = BytesIO(b'\x89PNG\x1a\n\x00\x00\x00IHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\xdac\x00\x01\x00\x00\x05\x00\x01\x0f\xa5\xe7\x10\x00\x00\x00\x00IEND\xaeB`\x82')
    files = {"file": ("corrupted.png", corrupted_image, "image/png")}
    
    response = client.post("/predict", files=files)
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Invalid image file" in response.json()["detail"]

def test_predict_endpoint_empty_file():
    """Test with empty file"""
    empty_file = BytesIO(b"")
    files = {"file": ("empty.jpg", empty_file, "image/jpeg")}
    
    response = client.post("/predict", files=files)
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Invalid image file" in response.json()["detail"]

def test_predict_endpoint_missing_file():
    """Test with missing file parameter"""
    response = client.post("/predict")
    
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "field required" in response.json()["detail"][0]["msg"]

def test_predict_endpoint_server_error():
    """Test handling of internal server errors"""
    # Mock the predict function to raise an exception
    with patch("api.endpoints.predict", side_effect=Exception("Test error")):
        files = {"file": open(TEST_IMAGE_PATH, "rb")}
        response = client.post("/predict", files=files)
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Error processing image" in response.json()["detail"]

def test_predict_response_model():
    """Test the response model structure"""
    # Mock prediction results
    mock_predictions = [
        {"class_label": "bird", "confidence": 0.8},
        {"class_label": "plane", "confidence": 0.2}
    ]
    
    with patch("api.endpoints.predict", return_value=mock_predictions):
        files = {"file": open(TEST_IMAGE_PATH, "rb")}
        response = client.post("/predict", files=files)
        
        data = response.json()
        
        # Check all required fields are present
        assert all(key in data for key in ["predictions", "processing_time", "timestamp"])
        
        # Check predictions structure
        assert isinstance(data["predictions"], list)
        assert all("class_label" in pred and "confidence" in pred for pred in data["predictions"])
        
        # Check processing_time is float
        assert isinstance(data["processing_time"], float)
        
        # Check timestamp is valid ISO format
        try:
            datetime.fromisoformat(data["timestamp"])
        except ValueError:
            pytest.fail("Timestamp is not in valid ISO format")

def test_predict_endpoint_multiple_files():
    """Test that endpoint handles multiple files gracefully (should still work with first file)"""
    files = [
        ("file", open(TEST_IMAGE_PATH, "rb")),
        ("file", open(TEST_IMAGE_PATH, "rb"))
    ]
    
    mock_predictions = [{"class_label": "test", "confidence": 1.0}]
    
    with patch("api.endpoints.predict", return_value=mock_predictions):
        response = client.post("/predict", files=files)
        
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["predictions"][0]["class_label"] == "test"

def test_predict_endpoint_logging(tmp_path, caplog):
    """Test that appropriate logging occurs"""
    # Set up a temporary log file
    log_file = tmp_path / "test.log"
    logging.basicConfig(filename=log_file, level=logging.INFO)
    
    mock_predictions = [{"class_label": "test", "confidence": 1.0}]
    
    with patch("api.endpoints.predict", return_value=mock_predictions):
        files = {"file": open(TEST_IMAGE_PATH, "rb")}
        response = client.post("/predict", files=files)
        
        assert response.status_code == status.HTTP_200_OK
        
        # Check logs were written
        with open(log_file, "r") as f:
            logs = f.read()
            
        assert "Processing image" in logs
        assert "Prediction completed" in logs
        assert "test_image.jpg" in logs