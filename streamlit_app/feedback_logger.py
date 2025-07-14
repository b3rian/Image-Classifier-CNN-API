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

    @classmethod
    def log_feedback(cls, predictions: list, positive: bool, correction: str = None):
        """
        Log user feedback to CSV and session state
        
        Args:
            predictions: List of prediction dicts from API
            positive: Whether feedback was positive
            correction: User-provided correct label (if negative feedback)
        """
        cls.init_feedback_file()
        
        # Get current prediction context
        top_pred = predictions[0]
        
        # Prepare data
        feedback_data = {
            "timestamp": datetime.now().isoformat(),
            "session_id": st.session_state.get('session_id', 'anonymous'),
            "model_version": cls.MODEL_VERSION,
            "predicted_class": top_pred['class_label'],
            "confidence": top_pred['confidence'],
            "user_feedback": "positive" if positive else "negative",
            "user_correction": correction
        }
        
        # Append to CSV
        with open(cls.FEEDBACK_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(feedback_data.values())
        
        # Also store in session for UI
        if 'feedback_history' not in st.session_state:
            st.session_state.feedback_history = []
        st.session_state.feedback_history.append(feedback_data)