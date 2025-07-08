import cv2

def preprocess_image(image_path, target_size=(640, 640)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)

    # CLAHE tăng tương phản
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Lọc nhiễu giữ cạnh
    image = cv2.bilateralFilter(image, 9, 75, 75)
    return image
