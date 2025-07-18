# ----------------------------
# 1. Use an official lightweight Python base image
# ----------------------------
    FROM python:3.10-slim

    # ----------------------------
    # 2. Set environment variables
    # ----------------------------
    
    # Prevents Python from writing .pyc files to disk
    ENV PYTHONDONTWRITEBYTECODE=1
    
    # Ensures output from Python is immediately flushed to the terminal
    ENV PYTHONUNBUFFERED=1
    
    # Streamlit disables CORS and sets the port to 7860 on Spaces
    ENV PORT=7860
    
    # ----------------------------
    # 3. Set the working directory inside the container
    # ----------------------------
    WORKDIR /app
    
    # ----------------------------
    # 4. Copy project files into the container
    # ----------------------------
    
    # First copy requirements to leverage Docker caching
    COPY requirements.txt .
    
    # Then copy the rest of the app
    COPY streamlit_app/ ./streamlit_app/
    
    # ----------------------------
    # 5. Install dependencies
    # ----------------------------
    
    # Install system dependencies for packages like numpy, pandas, etc.
    RUN apt-get update && apt-get install -y \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        wget \
        curl \
     && rm -rf /var/lib/apt/lists/*
    
    # Install Python dependencies
    RUN pip install --no-cache-dir -r requirements.txt
    
    # ----------------------------
    # 6. Expose the required port
    # ----------------------------
    EXPOSE 7860
    
    # ----------------------------
    # 7. Set Streamlit as the entrypoint
    # ----------------------------
    
    CMD ["streamlit", "run", "streamlit_app/main.py", "--server.port=7860", "--server.address=0.0.0.0"]
    