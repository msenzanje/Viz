from ultralytics import YOLO
import cv2

# capture device
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model and classes
model = YOLO("best_colab_model.pt")
yolo_classes = list(model.names.values())
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]


while True:
    success, img = cap.read()
    results = model(source=0, show=True, conf=.3)
    print(results)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(0) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

