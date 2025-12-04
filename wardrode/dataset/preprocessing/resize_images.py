import cv2
import os

DATASET_DIR = "dataset/"
PROCESSED_DIR = "dataset/processed/"

os.makedirs(PROCESSED_DIR, exist_ok=True)

# Loop through each category folder
for cls in os.listdir(DATASET_DIR):
    class_path = os.path.join(DATASET_DIR, cls)

    # ignore non-folders
    if not os.path.isdir(class_path) or cls in ["processed", "preprocessing", "raw"]:
        continue

    save_path = os.path.join(PROCESSED_DIR, cls)
    os.makedirs(save_path, exist_ok=True)

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            print("Skipped:", img_name)
            continue

        img = cv2.resize(img, (224, 224))
        cv2.imwrite(os.path.join(save_path, img_name), img)

print("All images resized and saved to dataset/processed/")
