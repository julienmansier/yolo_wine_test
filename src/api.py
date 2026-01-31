"""
FastAPI Wine Bottle Detection API

This API provides endpoints for detecting wine bottles in images using YOLO models.

Endpoints:
    POST /api/v1/detect - Detect wine bottles in a base64-encoded image
    GET /health - Health check endpoint
    GET /api/v1/info - Get API and model information

Usage:
    uvicorn src.api:app --host 0.0.0.0 --port 8000

    Or with reload for development:
    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
"""

import base64
import io
import os
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import numpy as np
from PIL import Image
import cv2

from src.utils.detector import WineBottleDetector


# Pydantic models for request/response
class DetectionRequest(BaseModel):
    """Request model for wine bottle detection."""
    image: str = Field(
        ...,
        description="Base64-encoded image (JPEG, PNG, etc.)",
        example="data:image/jpeg;base64,/9j/4AAQSkZJRg..."
    )
    confidence: Optional[float] = Field(
        0.25,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for detections (0.0-1.0)"
    )
    model: Optional[str] = Field(
        "n",
        description="YOLO model size: n (nano), s (small), m (medium), l (large), x (extra-large)",
        pattern="^[nsmxl]$"
    )
    return_annotated: Optional[bool] = Field(
        True,
        description="Whether to return the annotated image with bounding boxes"
    )

    @validator('image')
    def validate_base64(cls, v):
        """Validate that the image is valid base64."""
        try:
            # Handle data URI scheme
            if v.startswith('data:image'):
                v = v.split(',', 1)[1]

            # Try to decode
            base64.b64decode(v)
            return v
        except Exception as e:
            raise ValueError(f"Invalid base64 image data: {str(e)}")


class Detection(BaseModel):
    """Single detection result."""
    confidence: float = Field(..., description="Detection confidence score (0-1)")
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    class_name: str = Field(..., description="Detected class name")
    class_id: int = Field(..., description="COCO class ID")


class DetectionResponse(BaseModel):
    """Response model for wine bottle detection."""
    success: bool = Field(..., description="Whether the detection was successful")
    total_count: int = Field(..., description="Total number of wine bottles detected")
    detections: List[Detection] = Field(..., description="List of individual detections")
    model_used: str = Field(..., description="YOLO model used for detection")
    confidence_threshold: float = Field(..., description="Confidence threshold applied")
    annotated_image: Optional[str] = Field(
        None,
        description="Base64-encoded annotated image with bounding boxes (if requested)"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    version: str


class InfoResponse(BaseModel):
    """API information response."""
    service: str
    version: str
    description: str
    available_models: List[str]
    default_model: str
    default_confidence: float
    endpoints: List[str]


# Initialize FastAPI app
app = FastAPI(
    title="Wine Bottle Detection API",
    description="YOLO-based wine bottle detection service with configurable models and confidence thresholds",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# Global detector instance cache (one per model size)
detector_cache = {}


def get_detector(model_size: str = "n", confidence: float = 0.25) -> WineBottleDetector:
    """
    Get or create a detector instance for the specified model.

    Args:
        model_size: Model size (n/s/m/l/x)
        confidence: Confidence threshold

    Returns:
        WineBottleDetector instance
    """
    model_names = {
        'n': 'models/yolo26n.pt',
        's': 'models/yolo26s.pt',
        'm': 'models/yolo26m.pt',
        'l': 'models/yolo26l.pt',
        'x': 'models/yolo26x.pt',
    }

    model_name = model_names.get(model_size, 'yolo26n.pt')
    cache_key = f"{model_name}_{confidence}"

    if cache_key not in detector_cache:
        detector_cache[cache_key] = WineBottleDetector(
            model_name=model_name,
            confidence_threshold=confidence,
            include_vases=False  # Only wine bottles, no vases (--no-vases flag)
        )

    return detector_cache[cache_key]


def decode_image(base64_string: str) -> np.ndarray:
    """
    Decode a base64 string to a numpy array (OpenCV image).

    Args:
        base64_string: Base64-encoded image

    Returns:
        OpenCV image as numpy array
    """
    # Remove data URI prefix if present
    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(',', 1)[1]

    # Decode base64 to bytes
    image_bytes = base64.b64decode(base64_string)

    # Convert to PIL Image
    pil_image = Image.open(io.BytesIO(image_bytes))

    # Convert to OpenCV format (BGR)
    opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return opencv_image


def encode_image(image: np.ndarray) -> str:
    """
    Encode an OpenCV image to base64 string.

    Args:
        image: OpenCV image as numpy array

    Returns:
        Base64-encoded image string
    """
    # Encode image to JPEG
    success, buffer = cv2.imencode('.jpg', image)
    if not success:
        raise ValueError("Failed to encode image")

    # Convert to base64
    jpg_bytes = buffer.tobytes()
    base64_string = base64.b64encode(jpg_bytes).decode('utf-8')

    return f"data:image/jpeg;base64,{base64_string}"


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.

    Returns:
        Health status information
    """
    return HealthResponse(
        status="healthy",
        service="wine-bottle-detection-api",
        version="1.0.0"
    )


@app.get("/api/v1/info", response_model=InfoResponse)
async def get_info():
    """
    Get API information including available models and endpoints.

    Returns:
        API configuration and capabilities
    """
    return InfoResponse(
        service="Wine Bottle Detection API",
        version="1.0.0",
        description="YOLO-based wine bottle detection with configurable models",
        available_models=["n (nano)", "s (small)", "m (medium)", "l (large)", "x (extra-large)"],
        default_model="n (nano)",
        default_confidence=0.25,
        endpoints=["/health", "/api/v1/info", "/api/v1/detect"]
    )


@app.post("/api/v1/detect", response_model=DetectionResponse)
async def detect_bottles(request: DetectionRequest):
    """
    Detect wine bottles in a base64-encoded image.

    Args:
        request: DetectionRequest containing base64 image and parameters

    Returns:
        Detection results including count, bounding boxes, and optionally annotated image

    Example:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/detect" \\
             -H "Content-Type: application/json" \\
             -d '{
                   "image": "data:image/jpeg;base64,/9j/4AAQ...",
                   "confidence": 0.25,
                   "model": "n",
                   "return_annotated": true
                 }'
        ```
    """
    try:
        # Decode the image
        image = decode_image(request.image)

        # Save to temporary file for YOLO processing
        temp_dir = Path("/tmp/wine_bottle_api")
        temp_dir.mkdir(exist_ok=True)
        temp_image_path = temp_dir / "temp_input.jpg"
        cv2.imwrite(str(temp_image_path), image)

        # Get detector instance
        detector = get_detector(request.model, request.confidence)

        # Run detection
        count, detections = detector.detect_bottles(
            str(temp_image_path),
            visualize=False,
            save_path=None
        )

        # Format detections
        formatted_detections = [
            Detection(
                confidence=det['confidence'],
                bbox=det['bbox'],
                class_name=det['class'],
                class_id=det['class_id']
            )
            for det in detections
        ]

        # Generate annotated image if requested
        annotated_image_base64 = None
        if request.return_annotated:
            # Run detection again to get annotated image
            results = detector.model(str(temp_image_path), conf=request.confidence)
            annotated_image = results[0].plot()
            annotated_image_base64 = encode_image(annotated_image)

        # Clean up temp file
        temp_image_path.unlink(missing_ok=True)

        # Return response
        return DetectionResponse(
            success=True,
            total_count=count,
            detections=formatted_detections,
            model_used=f"yolo26{request.model}.pt",
            confidence_threshold=request.confidence,
            annotated_image=annotated_image_base64
        )

    except Exception as e:
        # Clean up temp file on error
        if 'temp_image_path' in locals():
            temp_image_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )


@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "service": "Wine Bottle Detection API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
        "endpoints": {
            "detect": "POST /api/v1/detect",
            "info": "GET /api/v1/info"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
