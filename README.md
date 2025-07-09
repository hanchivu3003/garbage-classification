# garbage-classification
garbage_yolov8/
├── data/
│   ├── raw/                      # Dữ liệu ảnh gốc từ Kaggle
│   │   ├── cardboard/
│   │   ├── glass/
│   │   ├── metal/
│   │   ├── paper/
│   │   ├── plastic/
│   │   └── trash/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
│
├── preprocess/
│   └── image_preprocessing.py   # Tiền xử lý ảnh (resize, cân bằng sáng, làm mượt)
│
├── scripts/
│   ├── convert_to_yolo.py       # Chuyển dữ liệu gốc sang định dạng YOLO
│   └── train_yolo.py            # Huấn luyện mô hình YOLOv8
│
├── dataset.yaml                 # Cấu hình dataset cho YOLO
└── README.md   