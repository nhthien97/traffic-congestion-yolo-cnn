from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def count_vehicles(image_path):
    results = model(image_path)[0]

    count = 0

    for box in results.boxes:
        cls = int(box.cls[0])

        if cls in [2, 3, 5, 7]:
            count += 1

    return count


if __name__ == "__main__":
    img = "bus.jpg"   # dùng ảnh bạn đã có
    print("Vehicles:", count_vehicles(img))