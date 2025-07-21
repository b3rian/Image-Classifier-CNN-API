# Dockerfile for Streamlit application using a lightweight Python image
FROM python:3.10-slim 

# Prevent Python from writing .pyc files to disk
ENV PYTHONDONTWRITEBYTECODE=1
# Force the stdout and stderr streams to be unbuffered
ENV PYTHONUNBUFFERED=1 

# Set working directory inside the container
WORKDIR /app

# Copy requirements file and application code to the container
COPY requirements.txt .
COPY streamlit_app/ ./streamlit_app/

# Install necessary system dependencies required by Streamlit and image processing libraries
RUN apt-get update && apt-get install -y \
    build-essential \             # Compilation tools
    libglib2.0-0 \                # Required by OpenCV
    libsm6 \                      # X11 session management library
    libxext6 \                    # X11 miscellaneous extensions
    libxrender-dev \             # Rendering support
    wget \                        # Tool for downloading files
    curl \                        # Used for health check probing
 && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make the main Streamlit script executable
RUN chmod a+x streamlit_app/main.py

# Expose the port Streamlit will run on
EXPOSE 7860

# Define a health check to ensure the container is running and responding
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:$PORT/ || exit 1

# Start the Streamlit application
CMD ["streamlit", "run", "streamlit_app/main.py", "--server.port=7860", "--server.address=0.0.0.0"]