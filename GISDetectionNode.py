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

class GISDetectionNode:
    """
    Combined GIS detection node that handles tiling and detection in one workflow.
    Optimized for 8GB VRAM systems and professional GIS standards.
    """
    
    def __init__(self):
        self.CATEGORY = "GIS"
        self.RETURN_TYPES = ("TENSOR", "METADATA")
        self.RETURN_NAMES = ("detection_output", "geo_metadata")
        
        # Memory constraints (in MB)
        self.MAX_VRAM = 7000  # Leave 1GB buffer for system
        self.MAX_RAM = 28000  # Leave 4GB buffer for system
        
    @classmethod
    def INPUT_TYPES(cls):
        """Define input types with memory-aware defaults"""
        # Get available models
        base_dir = os.path.join('models', 'ultralytics')
        all_models = []
        
        if os.path.exists(base_dir):
            for root, _, files in os.walk(base_dir):
                for file in files:
                    if file.endswith(('.pt', '.pth', '.onnx')):
                        rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                        model_name = os.path.splitext(rel_path)[0]
                        all_models.append(model_name)
        
        return {
            "required": {
                "input_path": ("STRING", {"default": ""}),
                "model_name": (sorted(all_models), {"default": all_models[0] if all_models else ""}),
                "input_type": (["single_image", "directory"], {"default": "single_image"}),
                "tile_scheme": (["SIMPLE", "MERCATOR", "UTM"], {"default": "SIMPLE"})
            },
            "optional": {
                # Processing options
                "use_parallel": ("BOOLEAN", {"default": True}),
                "max_parallel_images": ("INT", {"default": 4, "min": 1, "max": 16}),
                "confidence_threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0}),
                # Tiling options
                "tile_size": ("INT", {"default": 512, "min": 256, "max": 1024}),
                "tile_overlap": ("INT", {"default": 64, "min": 0, "max": 256}),
                "maintain_aspect": ("BOOLEAN", {"default": True}),
                # Memory options
                "memory_limit_mb": ("INT", {"default": 2048, "min": 1024, "max": 4096}),
                "image_extensions": ("STRING", {"default": ".tif,.tiff,.jpg,.png"})
            }
        }

    def detect_model_type(self, model_path):
        """Automatically detect model type from the model file"""
        try:
            model = torch.load(model_path, map_location='cpu')
            
            if hasattr(model, 'task'):
                task = model.task
                if task == 'detect':
                    return 'bbox'
                elif task in ['segment', 'segmentation']:
                    return 'segmentation'
                elif task == 'pose':
                    return 'keypoints'
                else:
                    return task
            
            return 'bbox'  # Default to bbox if cannot determine
            
        except Exception as e:
            print(f"Error detecting model type: {e}")
            return 'bbox'

    def create_tile_scheme(self, src, scheme_type, tile_size, tile_overlap):
        """Create appropriate tiling scheme based on input type"""
        scheme = {
            'type': scheme_type,
            'crs': src.crs.to_string(),
            'bounds': src.bounds,
            'transform': src.transform,
            'tile_size': tile_size,
            'overlap': tile_overlap,
            'original_size': (src.width, src.height)
        }
        
        if scheme_type == "MERCATOR":
            # Add Web Mercator specific info
            bounds = transform_bounds(src.crs, CRS.from_epsg(3857), *src.bounds)
            scheme['mercator_bounds'] = bounds
            scheme['zoom_level'] = self.calculate_zoom_level(bounds, tile_size)
            
        elif scheme_type == "UTM":
            # Add UTM zone specific info
            center_x = (src.bounds.left + src.bounds.right) / 2
            center_y = (src.bounds.bottom + src.bounds.top) / 2
            utm_zone = math.floor((center_x + 180) / 6) + 1
            hemisphere = 'N' if center_y >= 0 else 'S'
            scheme['utm_zone'] = utm_zone
            scheme['hemisphere'] = hemisphere
            
        return scheme

    def calculate_zoom_level(self, bounds, tile_size):
        """Calculate appropriate zoom level for mercator tiles"""
        x_min, y_min, x_max, y_max = bounds
        width = x_max - x_min
        resolution = width / tile_size
        zoom = math.floor(math.log2(40075016.686 / (resolution * 256)))
        return max(0, min(zoom, 20))

    def generate_tiles(self, src, scheme):
        """Generate tiles based on scheme"""
        tiles = []
        tile_size = scheme['tile_size']
        overlap = scheme['overlap']
        
        # Calculate effective tile size
        effective_size = tile_size - overlap
        
        # Calculate number of tiles
        n_tiles_y = math.ceil(src.height / effective_size)
        n_tiles_x = math.ceil(src.width / effective_size)
        
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                # Calculate tile bounds
                x_start = tx * effective_size
                y_start = ty * effective_size
                x_end = min(x_start + tile_size, src.width)
                y_end = min(y_start + tile_size, src.height)
                
                # Read tile data
                tile_data = src.read(window=((y_start, y_end), (x_start, x_end)))
                
                # Create tile metadata
                tile_meta = {
                    'tile_coords': (tx, ty),
                    'pixel_bounds': (x_start, y_start, x_end, y_end),
                    'geo_bounds': src.transform * (x_start, y_start, x_end, y_end),
                    'size': (y_end - y_start, x_end - x_start)
                }
                
                tiles.append((tile_data, tile_meta))
        
        return tiles

    def process_tile(self, tile_data, model_type):
        """Process individual tile data"""
        # Convert to tensor if needed
        if not isinstance(tile_data, torch.Tensor):
            tile_data = torch.from_numpy(tile_data)
        
        # Normalize and format
        if tile_data.max() > 1.0:
            tile_data = tile_data / 255.0
            
        # Ensure CHW format
        if tile_data.shape[0] not in [1, 3]:
            tile_data = tile_data.permute(2, 0, 1)
            
        # Add batch dimension if needed
        if len(tile_data.shape) == 3:
            tile_data = tile_data.unsqueeze(0)
        
        return tile_data

    def process_image(self, image_path, model_name, tile_scheme="SIMPLE", 
                     tile_size=512, tile_overlap=64, memory_limit_mb=2048):
        """Process a single image with tiling"""
        # Get model path and type
        model_path = os.path.join('models', 'ultralytics', model_name)
        if not os.path.exists(model_path):
            for ext in ['.pt', '.pth', '.onnx']:
                if os.path.exists(model_path + ext):
                    model_path += ext
                    break
        
        model_type = self.detect_model_type(model_path)
        
        with rasterio.open(image_path) as src:
            # Create tiling scheme
            scheme = self.create_tile_scheme(src, tile_scheme, tile_size, tile_overlap)
            
            # Generate tiles
            tiles = self.generate_tiles(src, scheme)
            
            # Process tiles
            processed_tiles = []
            tile_metadata = []
            
            for tile_data, tile_meta in tiles:
                processed_tile = self.process_tile(tile_data, model_type)
                processed_tiles.append(processed_tile)
                tile_metadata.append({
                    **tile_meta,
                    'model_type': model_type,
                    'scheme': scheme
                })
            
            # Stack tiles if needed
            if len(processed_tiles) > 1:
                processed_tiles = torch.cat(processed_tiles, dim=0)
            else:
                processed_tiles = processed_tiles[0]
            
            return processed_tiles, tile_metadata

    def get_optimal_workers(self, max_parallel_images):
        """Determine optimal number of worker processes"""
        available_cpu = cpu_count()
        available_ram = psutil.virtual_memory().available / (1024**2)  # MB
        
        ram_based_workers = int(available_ram / self.MAX_RAM)
        cpu_based_workers = max(1, available_cpu - 2)  # Leave 2 cores for system
        
        return min(max_parallel_images, ram_based_workers, cpu_based_workers)

    def get_image_paths(self, input_path, image_extensions):
        """Get list of image paths"""
        if os.path.isfile(input_path):
            return [input_path]
            
        image_paths = []
        extensions = image_extensions.split(',')
        for ext in extensions:
            image_paths.extend(list(Path(input_path).glob(f'*{ext.strip()}')))
        return sorted(image_paths)

    def execute(self, input_path, model_name, input_type="single_image", tile_scheme="SIMPLE",
                use_parallel=True, max_parallel_images=4, confidence_threshold=0.25,
                tile_size=512, tile_overlap=64, maintain_aspect=True,
                memory_limit_mb=2048, image_extensions=".tif,.tiff,.jpg,.png"):
        """Main execution method"""
        image_paths = self.get_image_paths(input_path, image_extensions)
        
        if not image_paths:
            raise ValueError(f"No images found in {input_path}")
        
        all_processed_data = []
        all_metadata = []
        
        if len(image_paths) == 1 or not use_parallel:
            # Process sequentially
            for img_path in image_paths:
                processed_data, metadata = self.process_image(
                    str(img_path), model_name, tile_scheme,
                    tile_size, tile_overlap, memory_limit_mb
                )
                all_processed_data.append(processed_data)
                all_metadata.append(metadata)
        else:
            # Process in parallel
            num_workers = self.get_optimal_workers(max_parallel_images)
            with Pool(num_workers) as pool:
                results = pool.starmap(
                    self.process_image,
                    [(str(path), model_name, tile_scheme, tile_size, 
                      tile_overlap, memory_limit_mb) for path in image_paths]
                )
                for processed_data, metadata in results:
                    all_processed_data.append(processed_data)
                    all_metadata.append(metadata)
        
        # Stack results if multiple images
        if len(all_processed_data) > 1:
            all_processed_data = torch.stack(all_processed_data)
        else:
            all_processed_data = all_processed_data[0]
            
        return (all_processed_data, all_metadata)

NODE_CLASS_MAPPINGS = {
    "GISDetectionNode": GISDetectionNode
}
