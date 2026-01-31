"""
Test script for Wine Bottle Detection API

This script demonstrates how to use the API with Python.
Run the API first with: uvicorn src.api:app --reload
"""

import base64
import sys
from pathlib import Path
import requests
import json


def test_api(image_path: str, model: str = "n", confidence: float = 0.25):
    """
    Test the wine bottle detection API with an image.

    Args:
        image_path: Path to the image file
        model: Model size (n/s/m/l/x)
        confidence: Confidence threshold
    """
    # Check if image exists
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        return

    print(f"Testing Wine Bottle Detection API")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Model: yolo26{model}.pt")
    print(f"Confidence: {confidence}")
    print()

    # Read and encode image
    print("1. Encoding image to base64...")
    with open(image_path, "rb") as f:
        image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    print(f"   Image size: {len(image_bytes):,} bytes")
    print(f"   Base64 size: {len(image_base64):,} characters")
    print()

    # Prepare request
    api_url = "http://localhost:8000/api/v1/detect"
    payload = {
        "image": f"data:image/jpeg;base64,{image_base64}",
        "confidence": confidence,
        "model": model,
        "return_annotated": True
    }

    # Send request
    print("2. Sending request to API...")
    print(f"   URL: {api_url}")
    try:
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API")
        print("   Make sure the API is running:")
        print("   uvicorn api:app --reload")
        return
    except requests.exceptions.Timeout:
        print("\n❌ Error: Request timed out")
        print("   Try using a smaller model or lower resolution image")
        return
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ Error: HTTP {response.status_code}")
        print(f"   {response.json().get('detail', 'Unknown error')}")
        return

    # Parse response
    result = response.json()
    print(f"   Status: {response.status_code} OK")
    print()

    # Display results
    print("3. Detection Results:")
    print("=" * 60)
    print(f"✓ Success: {result['success']}")
    print(f"✓ Total wine bottles detected: {result['total_count']}")
    print(f"✓ Model used: {result['model_used']}")
    print(f"✓ Confidence threshold: {result['confidence_threshold']}")
    print()

    if result['detections']:
        print("Detailed Detections:")
        print("-" * 60)
        for i, detection in enumerate(result['detections'], 1):
            print(f"\nBottle #{i}:")
            print(f"  Confidence: {detection['confidence']:.2%}")
            print(f"  Class: {detection['class_name']} (ID: {detection['class_id']})")
            bbox = detection['bbox']
            print(f"  Bounding Box: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
            print(f"  Size: {bbox[2]-bbox[0]:.1f}x{bbox[3]-bbox[1]:.1f} pixels")
    else:
        print("No wine bottles detected in the image.")

    print()

    # Save annotated image if available
    if result.get('annotated_image'):
        output_path = "test_api_output.jpg"
        print(f"4. Saving annotated image...")

        # Extract base64 data (remove data URI prefix if present)
        annotated_base64 = result['annotated_image']
        if annotated_base64.startswith('data:image'):
            annotated_base64 = annotated_base64.split(',', 1)[1]

        # Decode and save
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(annotated_base64))

        print(f"   ✓ Saved to: {output_path}")
    else:
        print("4. No annotated image returned")

    print()
    print("=" * 60)
    print("Test completed successfully!")
    print()


def test_health():
    """Test the health check endpoint."""
    print("Testing health check endpoint...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        result = response.json()
        print(f"✓ API Status: {result['status']}")
        print(f"✓ Service: {result['service']}")
        print(f"✓ Version: {result['version']}")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API")
        print("   Make sure the API is running:")
        print("   uvicorn api:app --reload")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_info():
    """Test the info endpoint."""
    print("\nGetting API information...")
    try:
        response = requests.get("http://localhost:8000/api/v1/info", timeout=5)
        result = response.json()
        print(f"✓ Service: {result['service']}")
        print(f"✓ Available models: {', '.join(result['available_models'])}")
        print(f"✓ Default model: {result['default_model']}")
        print(f"✓ Endpoints: {', '.join(result['endpoints'])}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Main test function."""
    print("\n" + "=" * 60)
    print("Wine Bottle Detection API - Test Script")
    print("=" * 60)
    print()

    # Test health and info endpoints first
    if not test_health():
        return 1

    if not test_info():
        return 1

    print()
    print("=" * 60)
    print()

    # Get image path from command line or prompt
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Usage: python test_api.py <image_path> [model] [confidence]")
        print("\nExample:")
        print("  python test_api.py sample_images/wine_bottle.jpg")
        print("  python test_api.py sample_images/wine_bottle.jpg m 0.20")
        print()
        image_path = input("Enter path to test image: ").strip()

    if not image_path:
        print("No image provided. Exiting.")
        return 1

    # Get model and confidence from command line
    model = sys.argv[2] if len(sys.argv) > 2 else "n"
    confidence = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25

    # Run detection test
    test_api(image_path, model, confidence)

    return 0


if __name__ == "__main__":
    exit(main())
