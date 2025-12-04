import cv2
import os
import shutil

PROCESSED_DIR = "dataset/processed/"
CLEAN_DIR = "dataset/processed_clean/"

os.makedirs(CLEAN_DIR, exist_ok=True)

def is_blurry(img_path, threshold=100):
    img = cv2.imread(img_path)
    if img is None:
        return True
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    focus = cv2.Laplacian(gray, cv2.CV_64F).var()
    return focus < threshold

for cls in os.listdir(PROCESSED_DIR):
    source_class = os.path.join(PROCESSED_DIR, cls)
    
    if not os.path.isdir(source_class):
        continue
    
    target_class = os.path.join(CLEAN_DIR, cls)
    os.makedirs(target_class, exist_ok=True)

    for img_name in os.listdir(source_class):
        img_path = os.path.join(source_class, img_name)

        if not is_blurry(img_path):
            shutil.copy(img_path, os.path.join(target_class, img_name))
        else:
            print("Removed blurry:", img_name)

print("✨ Blurry images removed. Clean images saved in dataset/processed_clean/")
