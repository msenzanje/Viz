import math
from ultralytics import YOLO
import random
import cv2
import numpy as np

imgpath = "image1.jpg"

model = YOLO("yolov8m-seg.pt")

img = cv2.imread(imgpath)
yolo_classes = list(model.names.values())
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]

conf = 0.5

results = model.predict(img, conf=conf)
colors = [random.choices(range(256), k=3) for _ in classes_ids]
print(results)

for result in results:
    for mask, box in zip(result.masks.xy, result.boxes):
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        points = np.int32([mask])
        color_number = classes_ids.index(int(box.cls[0]))

        cv2.polylines(img, points, True, colors[color_number], 5)

        org = [x1, y1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = .5
        color = (255, 255, 0)
        thickness = 2

        cls = int(box.cls[0])

        cv2.putText(img, yolo_classes[cls], org, font, fontScale, color, thickness)

cv2.imshow("Image", img)
cv2.waitKey(0)

cv2.imwrite("res", img)