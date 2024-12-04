from ultralytics import YOLO
from roboflow import Roboflow
from IPython import display
from IPython.display import display, Image

# # Base YOLOv8 segmentation model
# model = YOLO("yolov8n-seg.pt")

# Fine-tuned YOLOv8 model
model = YOLO("Augmented-Weights\\Session2\\best.pt")

# Importing Roboflow workspace
rf = Roboflow(api_key="UvVDJZVkAv5VXOa0jzUL")
project = rf.workspace("viz-kxwrm").project("sidewalk-detection-d54fs")

# Dowloading campus-specific dataset
version = project.version(11)
dataset = version.download("yolov8")

# Runs model in validation mode and stores results in cwd
model.val(data=f"{dataset.location}/data.yaml")

# Downloading third-party dataset
version = project.version(3)
dataset = version.download("yolov8")

# Runs model in validation mode and stores results in cwd
model.val(data=f"{dataset.location}/data.yaml")