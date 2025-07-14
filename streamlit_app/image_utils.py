import streamlit as st
import requests
import tempfile
import numpy as np
from PIL import Image, ImageOps
import io
import base64
import os
import json
from typing import List

# Constants
API_URL = "https://serverless.roboflow.com/infer/workflows/b3rian/custom-workflow"
API_KEY = "q8SYHtrPJTjKbgZLLSYf"
SUPPORTED_FORMATS = ["png", "jpg", "jpeg", "bmp"]
MODEL_OPTIONS = ["Roboflow-Model"]  # Roboflow only supports its own workflow
IMGBB_API_KEY = "b69dda7ca05b930e96803a145544107d"  # 🔑 Replace with your own Imgbb API Key

# Utility Functions
def compress_image(image: Image.Image, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def validate_image(file) -> bool:
    try:
        Image.open(file).verify()
        return True
    except:
        return False

def resize_image(image: Image.Image, max_dim: int = 480) -> Image.Image:
    image.thumbnail((max_dim, max_dim))
    return image

def fetch_image_from_url(url: str) -> Image.Image:
    response = requests.get(url)
    return Image.open(io.BytesIO(response.content))

def display_predictions(predictions):
    top = predictions[0]
    st.success(f"Top Prediction: {top['label']} ({top['confidence']}%)")
    st.progress(int(top['confidence']))
    with st.expander("See Top-5 Predictions"):
        for p in predictions:
            st.write(f"{p['label']}: {p['confidence']}%")

def upload_image_to_imgbb(image_bytes):
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "key": IMGBB_API_KEY,
        "image": encoded_image,
    }
    try:
        res = requests.post("https://api.imgbb.com/1/upload", data=payload)
        res.raise_for_status()
        return res.json()["data"]["url"]
    except Exception as e:
        st.error(f"Image upload failed: {e}")
        return None

def call_api(image_bytes):
    image_url = upload_image_to_imgbb(image_bytes)
    if not image_url:
        return None

    payload = {
        "api_key": API_KEY,
        "inputs": {
            "image": {
                "type": "url",
                "value": image_url
            }
        }
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API call failed: {e}")
        return None

def save_to_session(img_name, prediction):
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({"image": img_name, "prediction": prediction})

def download_link(data, filename, label):
    b64 = base64.b64encode(data.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">{label}</a>'

# Main App
st.set_page_config(page_title="Image Classifier", layout="wide")
st.title("🖼️ Smart Image Classifier Web App")

# Theme toggle
st.sidebar.title("Settings")
mode = st.sidebar.radio("Choose Mode", ["Upload", "Webcam", "URL"])
dark_mode = st.sidebar.checkbox("Dark Mode")

if dark_mode:
    st.markdown("""
        <style>
        body {
            background-color: #0e1117;
            color: #fafafa;
        }
        .stApp {
            background-color: #0e1117;
        }
        .css-1d391kg, .css-1v0mbdj {
            background-color: #0e1117;
            color: #fafafa;
        }
        .stButton>button {
            color: white;
            background-color: #333333;
        }
        .stTextInput>div>div>input, .stTextArea textarea {
            background-color: #333333;
            color: #fafafa;
        }
        </style>
    """, unsafe_allow_html=True)

# Upload Section
images = []
if mode == "Upload":
    uploaded_files = st.file_uploader(
        "Upload Images", type=SUPPORTED_FORMATS, accept_multiple_files=True
    )
    for file in uploaded_files:
        if validate_image(file):
            image = Image.open(file).convert("RGB")
            image = resize_image(image)
            images.append((file.name, image))
        else:
            st.error(f"File {file.name} is invalid or corrupted.")

elif mode == "Webcam":
    picture = st.camera_input("Take a Picture")
    if picture:
        image = Image.open(picture).convert("RGB")
        image = resize_image(image)
        images.append(("webcam.jpg", image))

elif mode == "URL":
    url = st.text_input("Enter Image URL")
    if url:
        try:
            image = fetch_image_from_url(url).convert("RGB")
            image = resize_image(image)
            images.append(("url_image.jpg", image))
        except:
            st.error("Could not load image from URL.")

# Preview & Prediction
if images:
    st.subheader("Preview & Adjust Images")
    for idx, (name, img) in enumerate(images):
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption=f"{name} ({img.size[0]}x{img.size[1]})")
        with col2:
            rotate_angle = st.slider(f"Rotate {name}", 0, 360, 0, key=name)
            if rotate_angle:
                img = img.rotate(rotate_angle)
                images[idx] = (name, img)

    if st.button("Classify Images"):
        for name, img in images:
            with st.spinner(f"Classifying {name}..."):
                compressed = compress_image(img)
                results = call_api(compressed)
                if results and "predictions" in results:
                    display_predictions(results["predictions"])
                    save_to_session(name, results["predictions"])

# Session History
st.sidebar.subheader("Session History")
if "history" in st.session_state and st.session_state.history:
    for entry in st.session_state.history[-5:][::-1]:
        st.sidebar.markdown(f"**{entry['image']}**")
        st.sidebar.markdown(f"Top: {entry['prediction'][0]['label']} ({entry['prediction'][0]['confidence']}%)")
    history_json = json.dumps(st.session_state.history)
    st.sidebar.markdown(download_link(history_json, "history.json", "🔍 Download History"), unsafe_allow_html=True)

# Feedback
st.subheader("📊 Feedback")
if st.session_state.get("history"):
    latest = st.session_state.history[-1]
    feedback_col1, feedback_col2 = st.columns([1, 3])
    with feedback_col1:
        st.write("Was this prediction accurate?")
        feedback = st.radio("Feedback", ["👍", "👎"], horizontal=True, key="feedback")
    with feedback_col2:
        comment = st.text_area("Comments or corrections? (optional)", key="comment")
        if st.button("Submit Feedback"):
            st.success("Thanks for your feedback!")

st.markdown("---")
st.caption("Image Classifier Web App | Built with 🚀 by B3rian")
