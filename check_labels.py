import os
import cv2
import numpy as np
from tqdm import tqdm

def read_label_yolo_obb(label_path):
    labels = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 9:
                labels.append([int(parts[0])] + list(map(float, parts[1:])))
    return labels

def draw_obb_labels(img, labels, color=(0, 255, 0), thickness=2):
    h, w = img.shape[:2]
    for label in labels:
        class_id = label[0]
        points = np.array(label[1:]).reshape(4, 2)
        pts = (points * np.array([w, h])).astype(np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
        cv2.putText(img, str(class_id), tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img

def draw_labels_on_folder(input_img_dir, input_label_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_img_dir) if f.endswith(('.jpg', '.png'))]

    for img_name in tqdm(image_files, desc="Drawing labels"):
        img_path = os.path.join(input_img_dir, img_name)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(input_label_dir, label_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        if os.path.exists(label_path):
            labels = read_label_yolo_obb(label_path)
            img = draw_obb_labels(img, labels)

        out_path = os.path.join(output_dir, img_name)
        cv2.imwrite(out_path, img)

    print(f"Finished drawing labels. Output saved in {output_dir}")

if __name__ == '__main__':
    input_img_folder = "dior_fewshot_data/finetune_combined_oversample/images"
    input_label_folder = "dior_fewshot_data/finetune_combined_oversample/labels"
    output_folder = "dior_fewshot_data/check_labels"

    draw_labels_on_folder(input_img_folder, input_label_folder, output_folder)
