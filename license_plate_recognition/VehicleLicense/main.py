from paddleocr import PaddleOCR
import cv2
import os

def load_images_from_folder(folder):
    imags = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                imags.append(img)

    return imags

ocr = PaddleOCR(use_angle_cls=True, use_gpu=False, over_version='PP-OCRv3')

# data
images = load_images_from_folder('D:\Project\license plate recognition\VehicleLicense\VehicleLicense\Infer')
test = []
for img in images:
    test.append(ocr.ocr(img, cls=True))
for i in test:
    print(i)
