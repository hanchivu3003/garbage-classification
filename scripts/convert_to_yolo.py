import os
import cv2
import shutil
from sklearn.model_selection import train_test_split

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
label_map = {name: i for i, name in enumerate(CLASSES)}

src_dir = 'data/raw'
img_out_dir = 'data/images'
label_out_dir = 'data/labels'

# Chuẩn bị thư mục
for split in ['train', 'val']:
    os.makedirs(f"{img_out_dir}/{split}", exist_ok=True)
    os.makedirs(f"{label_out_dir}/{split}", exist_ok=True)

data = []
for cls in CLASSES:
    cls_path = os.path.join(src_dir, cls)
    images = [os.path.join(cls_path, f) for f in os.listdir(cls_path) if f.endswith(('.jpg', '.png'))]
    data += [(img, cls) for img in images]

train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

def process_split(data_split, split):
    for img_path, cls in data_split:
        filename = os.path.basename(img_path)
        label = label_map[cls]

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # Copy ảnh
        new_img_path = f"{img_out_dir}/{split}/{filename}"
        shutil.copy(img_path, new_img_path)

        # Tạo file label
        x_center, y_center, width, height = 0.5, 0.5, 1.0, 1.0
        label_path = f"{label_out_dir}/{split}/{filename.replace('.jpg', '.txt').replace('.png', '.txt')}"
        with open(label_path, 'w') as f:
            f.write(f"{label} {x_center} {y_center} {width} {height}\n")

process_split(train_data, 'train')
process_split(val_data, 'val')
