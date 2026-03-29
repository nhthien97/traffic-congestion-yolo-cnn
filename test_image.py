from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

results = model("https://ultralytics.com/images/bus.jpg")

img = results[0].plot()

cv2.imwrite("result.jpg", img)

print("Saved result.jpg")