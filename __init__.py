"""
ComfyUI YOLO Training and Inference Nodes with GIS Detection
"""

import sys
import os
import subprocess
import importlib.util

# Check and install required packages
required_packages = {
    'ultralytics': 'ultralytics',
    'rasterio': 'rasterio',
    'cv2': 'opencv-python',
    'pyproj': 'pyproj'
}

for module_name, package_name in required_packages.items():
    if importlib.util.find_spec(module_name) is None:
        print(f"Installing {package_name}...")
        try:
            # For portable ComfyUI, use the embedded python
            if "python_embeded" in sys.executable or "python_embedded" in sys.executable:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "--no-warn-script-location"])
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✓ {package_name} installed successfully")
        except Exception as e:
            print(f"Failed to install {package_name}: {e}")
            print(f"Please install manually: python_embeded/python.exe -m pip install {package_name}")

# Initialize node mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Import YOLO training/inference nodes
try:
    from .yolo_nodes import NODE_CLASS_MAPPINGS as YOLO_CLASS_MAPPINGS
    from .yolo_nodes import NODE_DISPLAY_NAME_MAPPINGS as YOLO_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(YOLO_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(YOLO_DISPLAY_MAPPINGS)
    print(f"✓ YOLO Nodes loaded: {list(YOLO_DISPLAY_MAPPINGS.values())}")
except ImportError as e:
    print(f"Error loading YOLO nodes: {e}")

# Import GIS detection node
try:
    from .GISDetectionNode import NODE_CLASS_MAPPINGS as GIS_CLASS_MAPPINGS
    from .GISDetectionNode import NODE_DISPLAY_NAME_MAPPINGS as GIS_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(GIS_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(GIS_DISPLAY_MAPPINGS)
    print(f"✓ GIS Detection Node loaded: {list(GIS_DISPLAY_MAPPINGS.values())}")
except ImportError as e:
    print(f"Error loading GIS detection node: {e}")

print(f"\n✓ Total nodes loaded: {len(NODE_CLASS_MAPPINGS)}")
print(f"  - {', '.join(NODE_DISPLAY_NAME_MAPPINGS.values())}")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
