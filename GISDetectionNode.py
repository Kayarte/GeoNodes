import torch
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from pyproj import CRS, Transformer
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
import psutil
import math
from ultralytics import YOLO
from torchvision.ops import nms
import cv2
from PIL import Image
import json

class GISDetectionNode:
    """
    Combined GIS detection node that handles tiling and detection in one workflow.
    Optimized for 8GB VRAM systems and professional GIS standards.
    """
    
    def __init__(self):
        self.CATEGORY = "GIS"
        self.RETURN_TYPES = ("STRING", "STRING", "STRING")
        self.RETURN_NAMES = ("annotated_image_path", "detections_json", "status")

        # Memory constraints (in MB)
        self.MAX_VRAM = 7000  # Leave 1GB buffer for system
        self.MAX_RAM = 28000  # Leave 4GB buffer for system

        self.model = None
        self.current_model_path = None
        
    @classmethod
    def INPUT_TYPES(cls):
        """Define input types with memory-aware defaults"""
        return {
            "required": {
                "input_image_path": ("STRING", {"default": ""}),
                "model_path": ("STRING", {"default": ""}),
                "tile_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 32}),
                "overlap_percent": ("INT", {"default": 50, "min": 0, "max": 75, "step": 5}),
                "confidence_threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "iou_threshold": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01}),
                "device": (["cuda", "cpu", "mps"], {"default": "cuda"}),
            },
            "optional": {
                "output_dir": ("STRING", {"default": "output"}),
                "save_tiles": ("BOOLEAN", {"default": False}),
                "draw_labels": ("BOOLEAN", {"default": True}),
                "draw_conf": ("BOOLEAN", {"default": True}),
            }
        }

    def load_model(self, model_path, device):
        """Load YOLO model with caching"""
        if self.model is None or self.current_model_path != model_path:
            print(f"Loading YOLO model from: {model_path}")

            # Check device availability
            if device == "cuda" and not torch.cuda.is_available():
                print("CUDA not available, falling back to CPU")
                device = "cpu"
            elif device == "mps" and not torch.backends.mps.is_available():
                print("MPS not available, falling back to CPU")
                device = "cpu"

            self.model = YOLO(model_path)
            self.current_model_path = model_path

            if device == "cuda":
                torch.cuda.empty_cache()
                print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")

        return self.model, device

    def generate_tiles_with_overlap(self, image, tile_size, overlap_percent):
        """Generate overlapping tiles from image with proper stride calculation"""
        height, width = image.shape[:2]

        # Calculate stride based on overlap percentage
        stride = int(tile_size * (1 - overlap_percent / 100))

        tiles = []

        # Generate tiles with overlap
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                # Calculate tile bounds
                x_end = min(x + tile_size, width)
                y_end = min(y + tile_size, height)

                # Adjust start if we're at the edge to maintain tile_size
                x_start = max(0, x_end - tile_size)
                y_start = max(0, y_end - tile_size)

                # Extract tile
                tile = image[y_start:y_end, x_start:x_end]

                # Store tile with metadata
                tile_info = {
                    'tile': tile,
                    'x_offset': x_start,
                    'y_offset': y_start,
                    'width': x_end - x_start,
                    'height': y_end - y_start
                }

                tiles.append(tile_info)

        print(f"Generated {len(tiles)} tiles from {width}x{height} image (tile_size={tile_size}, overlap={overlap_percent}%)")
        return tiles

    def run_detection_on_tile(self, model, tile, x_offset, y_offset, conf_threshold, device):
        """Run YOLO detection on a single tile and transform coordinates to global space"""
        # Run YOLO inference
        results = model(tile, conf=conf_threshold, device=device, verbose=False)

        detections = []

        if len(results) > 0:
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes

                for i in range(len(boxes)):
                    # Get box coordinates (xyxy format)
                    box_xyxy = boxes.xyxy[i].cpu().numpy()

                    # Transform to global coordinates
                    x1_global = box_xyxy[0] + x_offset
                    y1_global = box_xyxy[1] + y_offset
                    x2_global = box_xyxy[2] + x_offset
                    y2_global = box_xyxy[3] + y_offset

                    detection = {
                        'bbox': [float(x1_global), float(y1_global), float(x2_global), float(y2_global)],
                        'confidence': float(boxes.conf[i].cpu().numpy()),
                        'class_id': int(boxes.cls[i].cpu().numpy()),
                        'class_name': result.names[int(boxes.cls[i])] if hasattr(result, 'names') else str(int(boxes.cls[i]))
                    }

                    detections.append(detection)

        return detections

    def apply_nms_to_detections(self, detections, iou_threshold):
        """Apply Non-Maximum Suppression to remove duplicate detections from overlapping tiles"""
        if len(detections) == 0:
            return []

        # Convert to tensors for NMS
        boxes = torch.tensor([d['bbox'] for d in detections], dtype=torch.float32)
        scores = torch.tensor([d['confidence'] for d in detections], dtype=torch.float32)

        # Apply NMS
        keep_indices = nms(boxes, scores, iou_threshold)

        # Filter detections
        filtered_detections = [detections[i] for i in keep_indices.tolist()]

        print(f"NMS: {len(detections)} detections -> {len(filtered_detections)} after merging (IoU threshold={iou_threshold})")

        return filtered_detections

    def draw_detections_on_image(self, image, detections, draw_labels=True, draw_conf=True):
        """Draw bounding boxes and labels on the image"""
        annotated_image = image.copy()

        # Generate colors for different classes
        class_ids = set(d['class_id'] for d in detections)
        colors = {}
        for class_id in class_ids:
            # Generate a unique color for each class
            np.random.seed(class_id)
            colors[class_id] = tuple(np.random.randint(0, 255, 3).tolist())

        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            class_id = detection['class_id']
            class_name = detection['class_name']
            confidence = detection['confidence']

            # Draw bounding box
            color = colors[class_id]
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

            # Draw label
            if draw_labels or draw_conf:
                label_parts = []
                if draw_labels:
                    label_parts.append(class_name)
                if draw_conf:
                    label_parts.append(f"{confidence:.2f}")

                label = " ".join(label_parts)

                # Get label size
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )

                # Draw label background
                cv2.rectangle(
                    annotated_image,
                    (x1, y1 - label_height - baseline - 5),
                    (x1 + label_width, y1),
                    color,
                    -1
                )

                # Draw label text
                cv2.putText(
                    annotated_image,
                    label,
                    (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

        return annotated_image

    def run(self, input_image_path, model_path, tile_size=512, overlap_percent=50,
            confidence_threshold=0.25, iou_threshold=0.45, device="cuda",
            output_dir="output", save_tiles=False, draw_labels=True, draw_conf=True):
        """Main execution method for GIS detection pipeline"""

        try:
            # Validate inputs
            if not os.path.exists(input_image_path):
                raise ValueError(f"Input image not found: {input_image_path}")
            if not os.path.exists(model_path):
                raise ValueError(f"Model not found: {model_path}")

            # Load model
            model, device = self.load_model(model_path, device)

            # Load image (support both regular images and GeoTIFF)
            try:
                # Try loading as GeoTIFF first
                with rasterio.open(input_image_path) as src:
                    image = src.read()
                    # Convert from (C, H, W) to (H, W, C)
                    if image.shape[0] in [1, 3, 4]:
                        image = np.transpose(image, (1, 2, 0))
                    # Handle multi-band images
                    if image.shape[2] > 3:
                        image = image[:, :, :3]
                    # Normalize if needed
                    if image.max() > 255:
                        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
            except:
                # Fall back to regular image loading
                image = cv2.imread(input_image_path)
                if image is None:
                    raise ValueError(f"Failed to load image: {input_image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            print(f"Loaded image: {image.shape}")

            # Generate tiles
            tiles = self.generate_tiles_with_overlap(image, tile_size, overlap_percent)

            # Run detection on all tiles
            all_detections = []
            for i, tile_info in enumerate(tiles):
                tile_detections = self.run_detection_on_tile(
                    model,
                    tile_info['tile'],
                    tile_info['x_offset'],
                    tile_info['y_offset'],
                    confidence_threshold,
                    device
                )
                all_detections.extend(tile_detections)

                # Optionally save tiles
                if save_tiles:
                    os.makedirs(output_dir, exist_ok=True)
                    tile_path = os.path.join(output_dir, f"tile_{i:04d}.jpg")
                    cv2.imwrite(tile_path, cv2.cvtColor(tile_info['tile'], cv2.COLOR_RGB2BGR))

            print(f"Total detections before NMS: {len(all_detections)}")

            # Apply NMS to remove duplicates
            filtered_detections = self.apply_nms_to_detections(all_detections, iou_threshold)

            # Draw detections on original image
            annotated_image = self.draw_detections_on_image(
                image, filtered_detections, draw_labels, draw_conf
            )

            # Save annotated image
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(input_image_path))[0]
            output_image_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
            cv2.imwrite(output_image_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

            # Save detections as JSON
            detections_json_path = os.path.join(output_dir, f"{base_name}_detections.json")
            with open(detections_json_path, 'w') as f:
                json.dump({
                    'image': input_image_path,
                    'image_size': {'width': image.shape[1], 'height': image.shape[0]},
                    'model': model_path,
                    'tile_size': tile_size,
                    'overlap_percent': overlap_percent,
                    'confidence_threshold': confidence_threshold,
                    'iou_threshold': iou_threshold,
                    'num_detections': len(filtered_detections),
                    'detections': filtered_detections
                }, f, indent=2)

            # Create status message
            status = f"Success: {len(filtered_detections)} detections found"
            print(f"\n{status}")
            print(f"Annotated image saved to: {output_image_path}")
            print(f"Detections JSON saved to: {detections_json_path}")

            # Clean up
            if device == "cuda":
                torch.cuda.empty_cache()

            return (output_image_path, detections_json_path, status)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return ("", "", error_msg)

    FUNCTION = "run"
    CATEGORY = "GIS/YOLO"

NODE_CLASS_MAPPINGS = {
    "GISDetectionNode": GISDetectionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GISDetectionNode": "GIS YOLO Detection (Tiled)"
}
