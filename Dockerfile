# -------- Stage 1: Base Image --------
    FROM python:3.10-slim

    # Set environment variables
    ENV PYTHONDONTWRITEBYTECODE=1
    ENV PYTHONUNBUFFERED=1
    
    # Set working directory
    WORKDIR /app
    
    # Install system dependencies
    RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        && rm -rf /var/lib/apt/lists/*
    
    # Install Python dependencies
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    
    # Copy FastAPI app source
    COPY api_backend/ ./api_backend/
    
    # Set the entry point
    CMD ["uvicorn", "api_backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
    