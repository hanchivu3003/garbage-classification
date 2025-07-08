from ultralytics import YOLO

# Load mô hình
model = YOLO("yolov8n.pt")  # hoặc yolov8s.yaml

# Huấn luyện
model.train(
    data="dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)
