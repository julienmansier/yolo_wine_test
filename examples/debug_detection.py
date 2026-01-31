"""
Debug Detection Script

This script shows all objects detected by YOLO in an image,
not just bottles and vases. Useful for understanding what
the model is seeing.
"""

import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))

from ultralytics import YOLO


def main():
    """Show all detections in an image."""

    # Get image path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter path to image: ").strip()

    if not image_path:
        print("No image path provided")
        return 1

    print(f"\nAnalyzing: {image_path}")
    print("=" * 70)

    # Load model
    model = YOLO('models/yolo26n.pt')

    # Run detection
    results = model(image_path, conf=0.25)

    # Show all detections
    for result in results:
        boxes = result.boxes
        print(f"\nTotal objects detected: {len(boxes)}")
        print()

        if len(boxes) == 0:
            print("No objects detected!")
            return 0

        # Group by class
        class_counts = {}
        detections_by_class = {}

        for box in boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy().tolist()

            if class_name not in class_counts:
                class_counts[class_name] = 0
                detections_by_class[class_name] = []

            class_counts[class_name] += 1
            detections_by_class[class_name].append({
                'confidence': confidence,
                'bbox': bbox,
                'class_id': class_id
            })

        # Print summary
        print("Detected objects by class:")
        print("-" * 70)
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {count}")
        print()

        # Print details
        print("Detailed detections:")
        print("-" * 70)
        for class_name in sorted(detections_by_class.keys()):
            dets = detections_by_class[class_name]
            print(f"\n{class_name.upper()} (Class ID: {dets[0]['class_id']}):")
            for i, det in enumerate(dets, 1):
                print(f"  #{i}: Confidence: {det['confidence']:.2%}")
                print(f"      BBox: [{det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}, {det['bbox'][2]:.1f}, {det['bbox'][3]:.1f}]")

        # Save annotated image
        annotated = result.plot()
        output_path = "debug_detection.jpg"

        import cv2
        cv2.imwrite(output_path, annotated)
        print(f"\nAnnotated image saved to: {output_path}")

    print("\n" + "=" * 70)
    print("Analysis complete!")

    # Print helpful info
    print("\nNote:")
    print("  - Class 39 = bottle")
    print("  - Class 75 = vase (wine bottles often detected as vases)")
    print("  - To exclude vases, use: include_vases=False in WineBottleDetector")

    return 0


if __name__ == "__main__":
    exit(main())
