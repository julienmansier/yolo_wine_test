"""
Batch Wine Bottle Detection Example

This script demonstrates how to process multiple images at once
and export results to CSV for further analysis.

Usage:
    python batch_processing.py <directory> [--model MODEL] [--conf THRESHOLD]

Arguments:
    directory: Path to directory containing images (optional, will prompt if not provided)
    --model MODEL: YOLO model to use (n/s/m/l/x, default: n for nano)
    --conf THRESHOLD: Confidence threshold (0-1, default: 0.25)

Examples:
    python batch_processing.py sample_images/
    python batch_processing.py sample_images/ --model m --conf 0.20
"""

import sys
from pathlib import Path
import glob
import pandas as pd

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))

from utils.detector import WineBottleDetector


def main():
    """Run batch detection on multiple images."""

    # Get directory with images - check command line arguments first
    if len(sys.argv) > 1:
        images_dir = sys.argv[1]
    else:
        images_dir = input("Enter directory path with wine bottle images (or press Enter for sample_images/): ").strip()

        if not images_dir:
            images_dir = "../sample_images"

    print(f"\nProcessing images from: {images_dir}")
    print("=" * 50)

    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(f"{images_dir}/{ext}"))

    if not image_paths:
        print(f"No images found in {images_dir}")
        print("Supported formats: JPG, JPEG, PNG")
        return 1

    print(f"Found {len(image_paths)} images to process\n")

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
        'n': 'yolo26n.pt',  # Nano - fastest, smallest
        's': 'yolo26s.pt',  # Small - balanced
        'm': 'yolo26m.pt',  # Medium - better accuracy
        'l': 'yolo26l.pt',  # Large - high accuracy
        'x': 'yolo26x.pt',  # Extra-large - best accuracy
    }
    model_name = model_names[model_size]

    # Initialize detector
    print(f"Loading YOLO model ({model_name})...")
    detector = WineBottleDetector(
        model_name=model_name,
        confidence_threshold=confidence_threshold
    )

    if model_size != 'n':
        print(f"Note: Using YOLO26{model_size.upper()} model for better accuracy")
    if confidence_threshold != 0.25:
        print(f"Note: Using custom confidence threshold: {confidence_threshold}")

    # Create output directory for annotated images
    output_dir = "batch_results"
    print(f"Results will be saved to: {output_dir}/\n")

    # Process all images
    print("Processing images...")
    results = detector.batch_detect(image_paths, output_dir=output_dir)

    # Prepare data for export
    data = []
    total_bottles = 0

    for img_path, (count, detections) in results.items():
        total_bottles += count
        data.append({
            'image': Path(img_path).name,
            'bottle_count': count,
            'detections': len(detections),
            'path': img_path
        })

        # Add individual detection details
        for i, det in enumerate(detections):
            data.append({
                'image': Path(img_path).name,
                'bottle_number': i + 1,
                'confidence': det['confidence'],
                'bbox_x1': det['bbox'][0],
                'bbox_y1': det['bbox'][1],
                'bbox_x2': det['bbox'][2],
                'bbox_y2': det['bbox'][3]
            })

    # Export results to CSV
    if data:
        df = pd.DataFrame(data)
        csv_path = f"{output_dir}/detection_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults exported to: {csv_path}")

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Images processed: {len(image_paths)}")
    print(f"Total bottles found: {total_bottles}")
    print(f"Average bottles per image: {total_bottles / len(image_paths):.1f}")

    # Show per-image breakdown
    print("\nPer-image breakdown:")
    for img_path, (count, _) in sorted(results.items(), key=lambda x: x[1][0], reverse=True):
        print(f"  {Path(img_path).name}: {count} bottles")

    return 0


if __name__ == "__main__":
    exit(main())
