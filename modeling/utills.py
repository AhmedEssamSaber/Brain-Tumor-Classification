# utils.py
import os
import os
from glob import glob

def load_image_paths_and_labels(base_path):
    image_paths = []
    labels = []

    for class_name in ['Brain Tumor', 'Healthy']:
        class_folder = os.path.join(base_path, class_name)
        class_label = 1 if class_name == 'Brain Tumor' else 0

        # Read all images inside class_folder
        for img_path in glob(os.path.join(class_folder, '*')):
            image_paths.append(img_path)
            labels.append(class_label)
    
    return image_paths, labels