"""
YOLO Model Comparison Example

This script compares different YOLO models (v8, v11, v26) on the same image
to help you choose the best model for your use case.

Compares:
- YOLO26n (latest nano - optimized for CPU, NMS-free)
- YOLO26s (latest small - balanced performance)
- YOLO11n (previous generation nano)
- YOLOv8n (mature, well-documented)
"""

import sys
from pathlib import Path
import time
import matplotlib.pyplot as plt

# Add parent directory to path to import src
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.detector import WineBottleDetector


def compare_models(image_path: str, models: list):
    """
    Compare multiple YOLO models on the same image.

    Args:
        image_path: Path to test image
        models: List of model names to compare

    Returns:
        Dictionary with comparison results
    """
    results = {}

    print(f"Testing image: {image_path}\n")

    for model_name in models:
        print(f"Testing {model_name}...")
        print("-" * 40)

        try:
            # Initialize detector
            detector = WineBottleDetector(
                model_name=model_name,
                confidence_threshold=0.25
            )

            # Time the detection
            start_time = time.time()
            count, detections = detector.detect_bottles(
                image_path,
                visualize=False,
                save_path=f"comparison_{model_name.replace('.pt', '')}.jpg"
            )
            elapsed_time = time.time() - start_time

            # Calculate average confidence
            avg_confidence = (
                sum(d['confidence'] for d in detections) / len(detections)
                if detections else 0
            )

            results[model_name] = {
                'count': count,
                'time': elapsed_time,
                'avg_confidence': avg_confidence,
                'detections': detections
            }

            print(f"  Bottles found: {count}")
            print(f"  Time: {elapsed_time:.3f}s")
            print(f"  Avg confidence: {avg_confidence:.2%}")
            print(f"  Speed: {1/elapsed_time:.1f} FPS")
            print()

        except Exception as e:
            print(f"  Error: {e}\n")
            results[model_name] = None

    return results


def visualize_comparison(results: dict):
    """Create visualization comparing model performance."""

    valid_results = {k: v for k, v in results.items() if v is not None}

    if not valid_results:
        print("No valid results to visualize")
        return

    models = list(valid_results.keys())
    counts = [valid_results[m]['count'] for m in models]
    times = [valid_results[m]['time'] for m in models]
    confidences = [valid_results[m]['avg_confidence'] * 100 for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Clean up model names for display
    display_names = [m.replace('.pt', '').upper() for m in models]

    # Plot 1: Detection Count
    axes[0].bar(display_names, counts, color='steelblue')
    axes[0].set_ylabel('Bottles Detected')
    axes[0].set_title('Detection Count by Model')
    axes[0].grid(axis='y', alpha=0.3)

    # Plot 2: Inference Time
    axes[1].bar(display_names, times, color='coral')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].set_title('Inference Speed by Model')
    axes[1].grid(axis='y', alpha=0.3)

    # Plot 3: Average Confidence
    axes[2].bar(display_names, confidences, color='mediumseagreen')
    axes[2].set_ylabel('Confidence (%)')
    axes[2].set_title('Average Confidence by Model')
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print("Comparison chart saved to: model_comparison.png")
    plt.show()


def main():
    """Run model comparison."""

    # Get test image - check command line arguments first
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter path to test image (or press Enter for sample): ").strip()

        if not image_path:
            image_path = "../sample_images/wine_bottles.jpg"

    print("\n" + "=" * 50)
    print("YOLO MODEL COMPARISON")
    print("=" * 50 + "\n")

    # Models to compare (from latest to previous generations)
    models_to_test = [
        'models/yolo26n.pt',  # Latest nano - 43% faster CPU, NMS-free
        'models/yolo26s.pt',  # Latest small - balanced performance
        'models/yolo11n.pt',  # Previous generation nano
        'models/yolov8n.pt',  # Mature, well-documented
    ]

    print("Models to compare:")
    for model in models_to_test:
        print(f"  - {model}")
    print()

    # Run comparison
    try:
        results = compare_models(image_path, models_to_test)

        # Print summary
        print("=" * 50)
        print("SUMMARY")
        print("=" * 50)

        valid_results = {k: v for k, v in results.items() if v is not None}

        if valid_results:
            # Find best models
            fastest = min(valid_results, key=lambda k: valid_results[k]['time'])
            most_confident = max(valid_results, key=lambda k: valid_results[k]['avg_confidence'])

            print(f"\nFastest model: {fastest} ({valid_results[fastest]['time']:.3f}s)")
            print(f"Most confident: {most_confident} ({valid_results[most_confident]['avg_confidence']:.2%})")

            # Visualize results
            print("\nGenerating comparison chart...")
            visualize_comparison(results)

        else:
            print("No successful detections to compare.")
            return 1

    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        print("\nTip: Place a test image in sample_images/ or provide a valid path")
        return 1

    except Exception as e:
        print(f"Error: {e}")
        return 1

    print("\nComparison complete!")
    return 0


if __name__ == "__main__":
    exit(main())
