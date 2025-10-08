"""
ComfyUI YOLO Training and Inference Nodes
"""

import sys
import os
import subprocess
import importlib.util

# Check if ultralytics is installed (the main package we need)
if importlib.util.find_spec('ultralytics') is None:
    print("Installing ultralytics for YOLO nodes...")
    try:
        # For portable ComfyUI, use the embedded python
        if "python_embeded" in sys.executable or "python_embedded" in sys.executable:
            # Portable version - use the embedded python's pip
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "--no-warn-script-location"])
        else:
            # Regular installation
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        print("✓ ultralytics installed successfully")
    except Exception as e:
        print(f"Failed to install ultralytics: {e}")
        print("Please install manually in ComfyUI's embedded python:")
        print("python_embeded/python.exe -m pip install ultralytics")

# Import the nodes
try:
    from .yolo_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    print(f"✓ YOLO Nodes loaded: {list(NODE_DISPLAY_NAME_MAPPINGS.values())}")
except ImportError as e:
    print(f"Error loading YOLO nodes: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']