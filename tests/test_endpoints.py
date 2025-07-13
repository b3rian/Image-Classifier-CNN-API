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