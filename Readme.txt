📁 Dataset Setup
Your dataset folder should look like this:
your_dataset_folder/
├── data.yaml           # Configuration file (REQUIRED)
├── train/
│   ├── images/        # Training images
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── labels/        # Training labels (text files)
│       ├── img001.txt
│       ├── img002.txt
│       └── ...
└── valid/             # Optional but recommended
    ├── images/        # Validation images
    └── labels/        # Validation labels

data.yaml Example:

path: .  # Path to dataset (. means current folder)
train: train/images  # Path to training images
val: valid/images    # Path to validation images (optional)

# Number of classes
nc: 3  # Change this to your number of classes

# Class names
names: ['person', 'car', 'dog']  # Change to your class names

🔧 The 3 Nodes
1️⃣ YOLO Train Config
What it does: Sets up all training settings.
Inputs:

model_size: Choose model (nano, small, medium, large, xlarge)
dataset_path: Path to your dataset folder
epochs: How long to train (default: 100)
batch_size: Images per batch (default: 16)
image_size: Image size for training (default: 640)
device: Use CPU or GPU (default: auto-detect)

Output: Configuration settings

2️⃣ YOLO Train
What it does: Actually trains your YOLO model.
Inputs:

config: Connect from YOLO Train Config node
resume: Continue from previous training? (yes/no)
pretrained: Use pretrained weights? (yes/no)

Outputs:

model_path: Where your trained model is saved
results: Training results and metrics

3️⃣ YOLO Detect
What it does: Use your trained model to detect objects in images.
Inputs:

model_path: Path to your trained model (from YOLO Train)
image: Image to detect objects in
confidence: Detection confidence (0.0-1.0, default: 0.25)
iou_threshold: Overlap threshold (0.0-1.0, default: 0.45)

Outputs:

image: Image with detection boxes drawn
detections: List of all detected objects




📖 Step-by-Step Guide
Step 1: Prepare Your Dataset

Create folder structure (see Dataset Setup above)
Add your images and labels
Create data.yaml file

Step 2: Configure Training

Add YOLO Train Config node
Set your model size (start with nano for testing)
Point to your dataset folder
Set epochs (50-100 is good to start)

Step 3: Train Model

Add YOLO Train node
Connect config output to it
Run the workflow
Wait for training to complete
Your model saves to runs/detect/train/weights/best.pt

Step 4: Test Detection

Add YOLO Detect node
Connect model_path from training
Load a test image
Run to see detections!