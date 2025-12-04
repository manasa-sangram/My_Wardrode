import cv2
import os
import numpy as np

DATA_DIR = "dataset/processed_clean/"

def load_data():
    images = []
    labels = []

    classes = sorted(os.listdir(DATA_DIR))
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    for cls in classes:
        class_path = os.path.join(DATA_DIR, cls)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0

            images.append(img)
            labels.append(class_to_idx[cls])

    return np.array(images), np.array(labels), class_to_idx
