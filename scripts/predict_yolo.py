from ultralytics import YOLO
import sys

# Đường dẫn tới mô hình đã huấn luyện
model_path = "runs/detect/train5/weights/best.pt"
img_path = "data/raw/cardboard/cardboard1.jpg"  # Mặc định

# Nếu truyền ảnh qua dòng lệnh
if len(sys.argv) > 1:
    img_path = sys.argv[1]

# Load mô hình đã huấn luyện
model = YOLO(model_path)

# Dự đoán trên ảnh
results = model(img_path)

# Lấy kết quả đầu tiên (vì chỉ có 1 ảnh)
result = results[0]

# Hiển thị kết quả và lưu ảnh
result.show()  # Hiển thị bounding box trên ảnh
result.save(filename="result.jpg")  # Lưu ảnh có bounding box

# In ra nhãn dự đoán và xác suất
for box in result.boxes:
    cls = int(box.cls[0])      # Lấy chỉ số lớp
    conf = float(box.conf[0])  # Xác suất
    print(f"Class: {model.names[cls]}, Confidence: {conf:.2f}")
