Viz

This project leverages computer vision and deep learning to help visually impaired individuals navigate 
sidewalks safely. Using YOLOv8, it detects and classifies objects like people, vehicles, and obstacles in 
real-time via a webcam feed, providing visual feedback.

Features
Real-Time Object Detection: Detects objects commonly encountered on sidewalks.
Efficient and Lightweight Model: Utilizes the YOLOv8n model for optimal speed and accuracy.
Customizable Classes: Supports a variety of object classes like pedestrians, vehicles, and benches.
Visual Feedback: Highlights detected objects and their labels on the video feed.
Technologies Used

Python
OpenCV: For image processing and webcam integration.
YOLOv8: For object detection and classification.
Prerequisites

Ensure the following are installed on your system:

Python 3.8+
Required Python libraries:
pip install ultralytics opencv-python
A webcam or external camera.
Installation and Setup

Clone this repository:
git clone https://github.com/msenzanje/Viz.git
cd sidewalk-navigator
Download YOLOv8 weights:
Save the pre-trained weights (yolov8n.pt) to the yolo-Weights/ directory.
Run the application:
python main.py
Usage

Start the script by running main.py.
The webcam feed will open, and objects detected in real-time will be highlighted with bounding boxes and labels.
Press q to exit the application.
Supported Object Classes

The following objects can be detected:

Pedestrians
Vehicles (cars, bicycles, trucks, etc.)
Traffic signs and lights
Common sidewalk objects (benches, potted plants, etc.)
Refer to the full list of YOLOv8 classes in the code for additional details.
Code Overview

Model Initialization:
The YOLOv8 model is loaded using the Ultralytics library.
Webcam Integration:
Captures frames in real-time using OpenCV.
Object Detection Pipeline:
Each frame is processed through the YOLOv8 model to identify objects.
Visual Feedback:
Bounding boxes and labels are overlaid on the webcam feed for detected objects.
