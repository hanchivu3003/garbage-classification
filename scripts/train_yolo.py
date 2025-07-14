from ultralytics import YOLO

if __name__ == '__main__':
    # Nếu cần, thêm: freeze_support()
    model = YOLO('yolov8n.pt')  # hoặc đường dẫn model
    model.train(
        data='dataset.yaml',
        epochs=50,
        imgsz=640,
        device=0  # dùng GPU
    )
