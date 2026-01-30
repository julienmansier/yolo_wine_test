"""
Batch Wine Bottle Detection Example

This script demonstrates how to process multiple images at once
and export results to CSV for further analysis.
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

    # Initialize detector
    print("Loading YOLO model...")
    detector = WineBottleDetector(
        model_name='yolo26n.pt',
        confidence_threshold=0.25
    )

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
