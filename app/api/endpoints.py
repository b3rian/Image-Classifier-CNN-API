import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from PIL import Image, UnidentifiedImageError
import numpy as np
import io
from typing import List, Dict
from pydantic import BaseModel
from datetime import datetime
import os

from app.predictor import predict
 

 