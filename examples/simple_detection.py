"""
Simple Wine Bottle Detection Example

This script demonstrates basic usage of the WineBottleDetector class
to detect and count wine bottles in a single image.

Usage:
    python simple_detection.py <image_path> [--model MODEL] [--no-vases] [--conf THRESHOLD] [--no-output]

Arguments:
    image_path: Path to the image file (optional, will prompt if not provided)
    --model MODEL: YOLO model to use (n/s/m/l/x, default: n for nano)
    --no-vases: Only count bottles (class 39), exclude vases (class 75)
    --conf THRESHOLD: Confidence threshold (0-1, default: 0.25)
    --no-output: Don't save annotated output image

Examples:
    python simple_detection.py sample_images/wine_bottle.jpeg
    python simple_detection.py sample_images/one_wine.jpg --no-vases
    python simple_detection.py sample_images/shelf_horizontal_close.jpg --model m --conf 0.20
    python simple_detection.py sample_images/wine_bottle.jpeg --no-output
"""

import sys
from pathlib import Path

# Add parent directory to path to import src
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.detector import WineBottleDetector


def main():
    """Run simple bottle detection on a sample image."""

    # Path to your test image - check command line arguments first
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter path to wine bottle image (or press Enter for sample): ").strip()

        if not image_path:
            print("No image path provided. Using sample path...")
            image_path = "../sample_images/wine_bottles.jpg"

    # Check for --model flag
    model_size = 'n'  # default: nano
    if '--model' in sys.argv:
        try:
            model_index = sys.argv.index('--model')
            if model_index + 1 < len(sys.argv):
                model_size = sys.argv[model_index + 1].lower()
                if model_size not in ['n', 's', 'm', 'l', 'x']:
                    print("Error: Model must be one of: n (nano), s (small), m (medium), l (large), x (extra-large)")
                    return 1
        except (ValueError, IndexError):
            print("Error: Invalid model size. Use: --model n/s/m/l/x")
            return 1

    # Check for --no-vases flag
    include_vases = True
    if '--no-vases' in sys.argv:
        include_vases = False

    # Check for --no-output flag
    save_output = True
    if '--no-output' in sys.argv:
        save_output = False

    # Check for --conf flag
    confidence_threshold = 0.25
    if '--conf' in sys.argv:
        try:
            conf_index = sys.argv.index('--conf')
            if conf_index + 1 < len(sys.argv):
                confidence_threshold = float(sys.argv[conf_index + 1])
                if not 0 < confidence_threshold <= 1:
                    print("Error: Confidence threshold must be between 0 and 1")
                    return 1
        except (ValueError, IndexError):
            print("Error: Invalid confidence threshold. Use: --conf 0.25")
            return 1

    # Model mapping
    model_names = {
        'n': 'models/yolo26n.pt',  # Nano - fastest, smallest
        's': 'models/yolo26s.pt',  # Small - balanced
        'm': 'models/yolo26m.pt',  # Medium - better accuracy
        'l': 'models/yolo26l.pt',  # Large - high accuracy
        'x': 'models/yolo26x.pt',  # Extra-large - best accuracy
    }
    model_name = model_names[model_size]

    print(f"\nDetecting bottles in: {image_path}")
    print("=" * 50)

    # Initialize detector with selected YOLO26 model
    print(f"Loading YOLO model ({model_name})...")
    detector = WineBottleDetector(
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        include_vases=include_vases
    )

    if model_size != 'n':
        print(f"Note: Using YOLO26{model_size.upper()} model for better accuracy")
    if not include_vases:
        print("Note: Vases are excluded (only counting class 39 'bottle')")
    if confidence_threshold != 0.25:
        print(f"Note: Using custom confidence threshold: {confidence_threshold}")

    # Display model info
    info = detector.get_model_info()
    print(f"Model: {info['model_name']}")
    print(f"Confidence threshold: {info['confidence_threshold']}")
    print()

    # Detect bottles
    print("Running detection...")
    try:
        count, detections = detector.detect_bottles(
            image_path,
            visualize=False,  # Don't show window (use False for CLI environments)
            save_path="output_detection.jpg" if save_output else None
        )

        print(f"\nResults:")
        print(f"Found {count} wine bottles!")

        if detections:
            print("\nDetailed detections:")
            for i, det in enumerate(detections, 1):
                print(f"  Bottle {i}:")
                print(f"    Confidence: {det['confidence']:.2%}")
                print(f"    BBox: {det['bbox']}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTip: Place a test image in the sample_images/ folder")
        print("     or provide a valid path when prompted.")
        return 1

    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

    print("\nDetection complete!")
    if save_output:
        print("Annotated image saved to: output_detection.jpg")
    return 0


if __name__ == "__main__":
    exit(main())
