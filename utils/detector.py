"""
Wine Bottle Detector using YOLO models.

This module provides a reusable class for detecting and counting wine bottles
in images using various YOLO models.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np
from ultralytics import YOLO


class WineBottleDetector:
    """
    A detector for wine bottles using YOLO models.

    Attributes:
        model: The loaded YOLO model
        bottle_class_id: COCO class ID for bottles (39)
        vase_class_id: COCO class ID for vases (86) - wine bottles often detected as vases
        confidence_threshold: Minimum confidence for detections
    """

    BOTTLE_CLASS_ID = 39  # Bottle class in COCO dataset
    VASE_CLASS_ID = 75    # Vase class - wine bottles are often detected as vases

    def __init__(
        self,
        model_name: str = 'yolo26n.pt',
        confidence_threshold: float = 0.25,
        include_vases: bool = True
    ):
        """
        Initialize the detector with a YOLO model.

        Args:
            model_name: YOLO model name (e.g., 'yolo26n.pt', 'yolo26s.pt', 'yolo11n.pt', 'yolov8n.pt')
            confidence_threshold: Minimum confidence score for detections (0-1)
            include_vases: Whether to count vases as wine bottles (default: True)
        """
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.include_vases = include_vases
        self.model_name = model_name

    def detect_bottles(
        self,
        image_path: str,
        visualize: bool = False,
        save_path: Optional[str] = None
    ) -> Tuple[int, List[Dict]]:
        """
        Detect wine bottles in an image.

        Args:
            image_path: Path to the image file
            visualize: Whether to display the detection results
            save_path: Optional path to save the annotated image

        Returns:
            Tuple of (bottle_count, detections_list)
            where detections_list contains dicts with 'bbox', 'confidence', 'class'
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Run detection
        results = self.model(image_path, conf=self.confidence_threshold)

        # Extract bottle detections (optionally including vases)
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                if class_id == self.BOTTLE_CLASS_ID:
                    detections.append({
                        'bbox': box.xyxy[0].cpu().numpy().tolist(),
                        'confidence': float(box.conf[0]),
                        'class': 'bottle',
                        'class_id': class_id
                    })
                elif class_id == self.VASE_CLASS_ID and self.include_vases:
                    detections.append({
                        'bbox': box.xyxy[0].cpu().numpy().tolist(),
                        'confidence': float(box.conf[0]),
                        'class': 'vase (likely wine bottle)',
                        'class_id': class_id
                    })

        bottle_count = len(detections)

        # Visualize if requested
        if visualize or save_path:
            annotated_image = results[0].plot()

            if save_path:
                cv2.imwrite(save_path, annotated_image)
                print(f"Saved annotated image to: {save_path}")

            if visualize:
                cv2.imshow('Wine Bottle Detection', annotated_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        return bottle_count, detections

    def batch_detect(
        self,
        image_paths: List[str],
        output_dir: Optional[str] = None
    ) -> Dict[str, Tuple[int, List[Dict]]]:
        """
        Detect bottles in multiple images.

        Args:
            image_paths: List of image file paths
            output_dir: Optional directory to save annotated images

        Returns:
            Dictionary mapping image paths to (count, detections)
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        results = {}
        for img_path in image_paths:
            try:
                save_path = None
                if output_dir:
                    filename = Path(img_path).name
                    save_path = os.path.join(output_dir, f"detected_{filename}")

                count, detections = self.detect_bottles(
                    img_path,
                    visualize=False,
                    save_path=save_path
                )
                results[img_path] = (count, detections)
                print(f"{img_path}: Found {count} bottles")

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                results[img_path] = (0, [])

        return results

    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'confidence_threshold': self.confidence_threshold,
            'bottle_class_id': self.BOTTLE_CLASS_ID,
            'vase_class_id': self.VASE_CLASS_ID,
            'model_type': self.model.task,
            'note': 'Detects both bottles (class 39) and vases (class 86) as wine bottles are often misclassified as vases'
        }
