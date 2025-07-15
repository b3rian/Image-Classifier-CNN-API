from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import time
import uvicorn

# =================== Config ===================
MODEL_PATH = "cnn_model.keras"
CLASS_NAMES = ["cat", "dog", "bird", "other"]  # Class labels
IMAGE_SIZE = (480, 480)  # Match model input size

# =================== Load Model ===================
print("🔄 Loading model...")
model = load_model(MODEL_PATH)
inference_model = tf.function(model)  # Optimized inference
print("✅ Model loaded successfully.")