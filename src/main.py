from PIL import Image
import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO
import torchvision.transforms as transforms

# ===== CNN MODEL =====
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3),
            nn.ReLU(),
            nn.Conv2d(32,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*48*48,128),
            nn.ReLU(),
            nn.Linear(128,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.fc(self.conv(x))

# ===== LOAD MODEL =====
model_cnn = CNN()
model_cnn.load_state_dict(torch.load("models/cnn.pth"))
model_cnn.eval()

model_yolo = YOLO("yolov8n.pt")

# ===== TRANSFORM =====
transform = transforms.Compose([
    transforms.Resize((100,100)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

# ===== VIDEO =====
cap = cv2.VideoCapture("traffic.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ===== YOLO =====
    results = model_yolo(frame)[0]

    vehicle_count = 0
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls in [2,3,5,7]:
            vehicle_count += 1

    # ===== CNN =====
    img = cv2.resize(frame, (100,100))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    pil_img = Image.fromarray(gray)

    tensor = transform(pil_img).unsqueeze(0)
    pred = model_cnn(tensor)

    label = "JAM" if pred.item() > 0.5 else "NORMAL"

    # ===== HIỂN THỊ =====
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f"Status: {label}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Traffic", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()