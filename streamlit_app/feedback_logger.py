import csv
import json
import time
from pathlib import Path
from datetime import datetime
import streamlit as st
from config import config

class FeedbackLogger:
    FEEDBACK_FILE = Path("feedback/classification_feedback.csv")
    MODEL_VERSION = "EfficientNetV2L-1.0"

    @classmethod
    def init_feedback_file(cls):
        """Initialize feedback file with headers if needed"""
        cls.FEEDBACK_FILE.parent.mkdir(exist_ok=True)
        if not cls.FEEDBACK_FILE.exists():
            with open(cls.FEEDBACK_FILE, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "session_id",
                    "model_version",
                    "predicted_class",
                    "confidence",
                    "user_feedback",
                    "user_correction"
                ])
