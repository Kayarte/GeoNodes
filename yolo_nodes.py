import torch
import os
import gc
import glob
from ultralytics import YOLO

class YOLOTrainNode:
    """
    ComfyUI node for training YOLO models (YOLO11, YOLOv8, etc.)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "model_size": (["n", "s", "m", "l", "x"], {
                    "default": "n"
                }),
                "task": (["detect", "segment", "classify", "pose", "obb"], {
                    "default": "detect"
                }),
                "epochs": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 10000,
                    "step": 1
                }),
                "batch_size": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 128,
                    "step": 1
                }),
                "img_size": ("INT", {
                    "default": 640,
                    "min": 32,
                    "max": 1280,
                    "step": 32
                }),
                "device": (["cuda", "cpu", "mps"], {
                    "default": "cuda"
                }),
                "pretrained": ("BOOLEAN", {
                    "default": True
                }),
                "learning_rate": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0001,
                    "max": 1.0,
                    "step": 0.0001
                }),
                "patience": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "val_split": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.05
                }),
            },
            "optional": {
                "project_name": ("STRING", {
                    "default": "yolo_training",
                    "multiline": False,
                }),
                "name": ("STRING", {
                    "default": "exp",
                    "multiline": False,
                }),
                "resume": ("BOOLEAN", {
                    "default": False
                }),
                "checkpoint_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("model_path", "results_dir", "status")
    FUNCTION = "train_model"
    CATEGORY = "YOLO"
    
    def train_model(self, dataset_path, model_size, task, epochs, batch_size, 
                    img_size, device, pretrained, learning_rate, patience, val_split,
                    project_name="yolo_training", name="exp", resume=False, checkpoint_path=""):
        
        try:
            # Validate dataset path
            if not os.path.exists(dataset_path):
                raise ValueError(f"Dataset path does not exist: {dataset_path}")
            
            # Check for data.yaml
            data_yaml = os.path.join(dataset_path, "data.yaml")
            if not os.path.exists(data_yaml):
                # Alternative: look for .yaml files in the dataset path
                yaml_files = glob.glob(os.path.join(dataset_path, "*.yaml"))
                if yaml_files:
                    data_yaml = yaml_files[0]
                    print(f"Using data config: {data_yaml}")
                else:
                    raise ValueError(f"No YAML configuration file found in {dataset_path}")
            
            # Construct model name based on task and pretrained status
            if resume and checkpoint_path and os.path.exists(checkpoint_path):
                print(f"Resuming from checkpoint: {checkpoint_path}")
                model = YOLO(checkpoint_path)
            else:
                if pretrained:
                    # Use pretrained weights
                    model_name = f"yolo11{model_size}"
                    if task == "segment":
                        model_name += "-seg"
                    elif task == "classify":
                        model_name += "-cls"
                    elif task == "pose":
                        model_name += "-pose"
                    elif task == "obb":
                        model_name += "-obb"
                    model_name += ".pt"
                else:
                    # Use configuration file (training from scratch)
                    print("Warning: Training from scratch is not recommended for most use cases")
                    model_name = f"yolo11{model_size}"
                    if task == "segment":
                        model_name += "-seg"
                    elif task == "classify":
                        model_name += "-cls"
                    elif task == "pose":
                        model_name += "-pose"
                    elif task == "obb":
                        model_name += "-obb"
                    model_name += ".yaml"
                
                print(f"Loading model: {model_name}")
                model = YOLO(model_name)
            
            # Check device availability
            if device == "cuda":
                if not torch.cuda.is_available():
                    print("CUDA not available, falling back to CPU")
                    device = "cpu"
                else:
                    # Clear CUDA cache before training
                    torch.cuda.empty_cache()
                    print(f"CUDA device available: {torch.cuda.get_device_name(0)}")
            elif device == "mps":
                if not torch.backends.mps.is_available():
                    print("MPS not available, falling back to CPU")
                    device = "cpu"
            
            # Prepare training arguments
            train_args = {
                "data": data_yaml,
                "epochs": epochs,
                "batch": batch_size,
                "imgsz": img_size,
                "device": device,
                "lr0": learning_rate,
                "patience": patience,
                "project": project_name,
                "name": name,
                "verbose": True,
                "exist_ok": True,  # Allow overwriting existing project
                "val": True,  # Enable validation
                "split": val_split,  # Validation split ratio
                "save": True,  # Save checkpoints
                "save_period": -1,  # Save best only
                "plots": True,  # Generate plots
            }
            
            # Add resume if specified
            if resume and not (checkpoint_path and os.path.exists(checkpoint_path)):
                train_args["resume"] = True
            
            # Train model
            print(f"Starting training on {device}...")
            print(f"Training parameters: {epochs} epochs, batch size {batch_size}, image size {img_size}")
            
            # Run training
            results = model.train(**train_args)
            
            # Determine paths
            results_dir = os.path.join(project_name, name)
            weights_dir = os.path.join(results_dir, "weights")
            
            # Find the best model
            best_model_path = os.path.join(weights_dir, "best.pt")
            last_model_path = os.path.join(weights_dir, "last.pt")
            
            # Check which model exists
            if os.path.exists(best_model_path):
                final_model_path = best_model_path
                status = "Training completed successfully (best.pt)"
            elif os.path.exists(last_model_path):
                final_model_path = last_model_path
                status = "Training completed (last.pt - no improvement found)"
            else:
                # Try to find any .pt file in weights directory
                pt_files = glob.glob(os.path.join(weights_dir, "*.pt"))
                if pt_files:
                    final_model_path = pt_files[0]
                    status = f"Training completed (using {os.path.basename(pt_files[0])})"
                else:
                    raise ValueError(f"No trained model found in {weights_dir}")
            
            print(f"Training complete! Model saved to: {final_model_path}")
            print(f"Results saved to: {results_dir}")
            
            # Get training metrics if available
            if hasattr(model, 'metrics'):
                metrics = model.metrics
                if metrics:
                    print(f"Final metrics: {metrics}")
            
            # Clean up memory
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            
            return (final_model_path, results_dir, status)
            
        except Exception as e:
            error_msg = f"Training error: {str(e)}"
            print(error_msg)
            
            # Clean up on error
            if 'model' in locals():
                del model
            gc.collect()
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return ("", "", error_msg)


class YOLOInferenceNode:
    """
    ComfyUI node for running inference with trained YOLO models
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "image_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "confidence": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "iou_threshold": ("FLOAT", {
                    "default": 0.45,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "device": (["cuda", "cpu", "mps"], {
                    "default": "cuda"
                }),
            },
            "optional": {
                "save_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "show_labels": ("BOOLEAN", {
                    "default": True
                }),
                "show_conf": ("BOOLEAN", {
                    "default": True
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("output_path", "detections_count", "results_json")
    FUNCTION = "run_inference"
    CATEGORY = "YOLO"
    
    def run_inference(self, model_path, image_path, confidence, iou_threshold, 
                     device, save_path="", show_labels=True, show_conf=True):
        
        try:
            import json
            
            # Validate inputs
            if not os.path.exists(model_path):
                raise ValueError(f"Model path does not exist: {model_path}")
            if not os.path.exists(image_path):
                raise ValueError(f"Image path does not exist: {image_path}")
            
            # Check device
            if device == "cuda" and not torch.cuda.is_available():
                print("CUDA not available, falling back to CPU")
                device = "cpu"
            elif device == "mps" and not torch.backends.mps.is_available():
                print("MPS not available, falling back to CPU")
                device = "cpu"
            
            # Load model
            print(f"Loading model from: {model_path}")
            model = YOLO(model_path)
            
            # Run inference
            print(f"Running inference on: {image_path}")
            results = model(
                image_path,
                conf=confidence,
                iou=iou_threshold,
                device=device,
                save=bool(save_path),
                show_labels=show_labels,
                show_conf=show_conf
            )
            
            # Process results
            result = results[0]
            detections_count = len(result.boxes) if result.boxes is not None else 0
            
            # Prepare output path
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                output_path = os.path.join(save_path, f"result_{os.path.basename(image_path)}")
                result.save(output_path)
            else:
                output_path = ""
            
            # Create results JSON
            results_data = {
                "image": image_path,
                "detections": detections_count,
                "boxes": []
            }
            
            if result.boxes is not None:
                for box in result.boxes:
                    box_data = {
                        "xyxy": box.xyxy.tolist()[0] if box.xyxy is not None else [],
                        "confidence": float(box.conf) if box.conf is not None else 0.0,
                        "class": int(box.cls) if box.cls is not None else -1,
                    }
                    if hasattr(result, 'names') and result.names:
                        box_data["class_name"] = result.names.get(int(box.cls), "unknown")
                    results_data["boxes"].append(box_data)
            
            results_json = json.dumps(results_data, indent=2)
            
            print(f"Inference complete! Found {detections_count} detections")
            
            # Clean up
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            
            return (output_path, detections_count, results_json)
            
        except Exception as e:
            error_msg = f"Inference error: {str(e)}"
            print(error_msg)
            
            # Clean up on error
            if 'model' in locals():
                del model
            gc.collect()
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return ("", 0, json.dumps({"error": error_msg}))


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "YOLOTrainNode": YOLOTrainNode,
    "YOLOInferenceNode": YOLOInferenceNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YOLOTrainNode": "YOLO Train",
    "YOLOInferenceNode": "YOLO Inference"
}