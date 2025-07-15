from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Callable
import numpy as np
from PIL import Image
import io
import time
import tensorflow as tf
import uvicorn

# Keras preprocessors and decoders
from tensorflow.keras.applications.efficientnet_v2 import (
    preprocess_input as efficientnet_preprocess,
    decode_predictions as efficientnet_decode
)
from tensorflow.keras.applications.resnet50 import (
    preprocess_input as resnet_preprocess,
    decode_predictions as resnet_decode
)