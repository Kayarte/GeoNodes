GeoDetect-ComfyUI
AI-powered object detection for geospatial imagery in ComfyUI. Process satellite, aerial, and drone imagery while preserving real-world coordinates.
What It Does
This node brings YOLO-style object detection to GIS workflows. It automatically tiles large geospatial images, runs detection models, and maintains geographic coordinates throughout - so you know exactly where objects were detected, not just what they are.
Perfect for:

🛰️ Detecting objects in satellite imagery (buildings, vehicles, infrastructure)
🚁 Processing drone surveys with geographic accuracy
🗺️ Creating detection heatmaps with real-world coordinates
📊 Counting features across large aerial images

Key Features

Smart Tiling: Automatically splits massive GeoTIFF files into processable chunks
Coordinate Preservation: Maintains geographic metadata (CRS, bounds, transforms)
Memory Optimized: Designed for 8GB VRAM systems
Multiple Tiling Schemes: Simple grid, Web Mercator, or UTM projections
Auto Model Detection: Supports bbox, segmentation, and keypoint models
Parallel Processing: Process multiple images simultaneously

How It Works

Input: Large geospatial image (GeoTIFF, etc.)
Tiling: Splits into overlapping tiles (default 512x512px)
Detection: Runs YOLO/Ultralytics models on each tile
Output: Detections with preserved geographic coordinates
