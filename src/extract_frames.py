import cv2
import os

video_path = "traffic.mp4"
output_folder = "dataset/jam"   # tạm để jam

cap = cv2.VideoCapture(video_path)

count = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if count % 10 == 0:
        filename = os.path.join(output_folder, f"frame_{saved}.jpg")
        cv2.imwrite(filename, frame)
        saved += 1

    count += 1

cap.release()

print("Saved frames:", saved)