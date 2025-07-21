# Use official Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create models directory and download models from HF
RUN mkdir -p /app/models && \
curl -fSL -o /app/models/efficientnet.keras https://huggingface.co/b3rian/resnet/resolve/main/efficientnet.keras && \
curl -fSL -o /app/models/resnet50_imagenet.keras https://huggingface.co/b3rian/resnet/resolve/main/resnet50_imagenet.keras

# Copy API code into the container directory
COPY api_backend/ ./api_backend/

# Expose the port the app runs on
EXPOSE 7860

# Set environment variables
ENV MODEL_DIR=/app/models

# Run the API
CMD ["uvicorn", "api_backend.main:app", "--host", "0.0.0.0", "--port", "7860"]