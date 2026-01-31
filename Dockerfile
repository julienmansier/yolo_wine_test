# Wine Bottle Detection API - Dockerfile
# Multi-stage build for optimized image size

# Stage 1: Base image with Python and dependencies
FROM python:3.9-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and health checks
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Application
FROM base AS app

# Copy application code
COPY src/ ./src/

# Copy YOLO model weights (at least nano model)
# You can add more models if needed
COPY models/ ./models/

# Create temp directory for API processing
RUN mkdir -p /tmp/wine_bottle_api

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
