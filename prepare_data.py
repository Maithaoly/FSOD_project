import os
import shutil
import xml.etree.ElementTree as ET
import yaml
import random
from collections import defaultdict
from tqdm import tqdm

IMAGE_DIR_TRAINVAL = 'DATA/dior/Images/trainval'
ANNOTATION_DIR_TRAINVAL = 'DATA/dior/Annotations/trainval'
FEW_SHOT_OUTPUT_DIR = 'dior_fewshot_data_v2'

ALL_IMAGE_FILES = sorted([f for f in os.listdir(IMAGE_DIR_TRAINVAL) if f.endswith(".jpg")])

K_SHOT = 10
IMG_WIDTH = 800
IMG_HEIGHT = 800
TRAIN_SET_SIZE = int(len(ALL_IMAGE_FILES) * 0.8)

BASE_CLASSES = [
    'airport', 'Expressway-Service-area', 'Expressway-toll-station',
    'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'stadium',
    'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill', 'basketballcourt'
]
NOVEL_CLASSES = ['baseballfield', 'airplane', 'bridge', 'chimney', 'ship']
ALL_CLASSES = sorted(BASE_CLASSES + NOVEL_CLASSES)

CLASS_MAPPING = {name: i for i, name in enumerate(ALL_CLASSES)}
BASE_CLASS_IDS = {CLASS_MAPPING.get(name) for name in BASE_CLASSES}
NOVEL_CLASS_IDS = {CLASS_MAPPING.get(name) for name in NOVEL_CLASSES}


def process_and_convert_labels(image_files, oimages_dir, olabels_dir, temp_labels_dir, class_mapping):
    image_to_classes = defaultdict(set)
    os.makedirs(temp_labels_dir, exist_ok=True)

    for image_file in tqdm(image_files, desc="Converting XML to TXT"):
        xml_path = os.path.join(olabels_dir, image_file.replace(".jpg", ".xml"))
        
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            label_path = os.path.join(temp_labels_dir, image_file.replace(".jpg", ".txt"))
            
            with open(label_path, "w") as label_file:
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    if class_name in class_mapping:
                        class_id = class_mapping[class_name]
                        coords = None
                        robndbox = obj.find('robndbox')
                        bndbox = obj.find('bndbox')

                        if robndbox is not None:
                            coords = [
                                float(robndbox.find('x_left_top').text) / IMG_WIDTH,
                                float(robndbox.find('y_left_top').text) / IMG_HEIGHT,
                                float(robndbox.find('x_right_top').text) / IMG_WIDTH,
                                float(robndbox.find('y_right_top').text) / IMG_HEIGHT,
                                float(robndbox.find('x_right_bottom').text) / IMG_WIDTH,
                                float(robndbox.find('y_right_bottom').text) / IMG_HEIGHT,
                                float(robndbox.find('x_left_bottom').text) / IMG_WIDTH,
                                float(robndbox.find('y_left_bottom').text) / IMG_HEIGHT,
                            ]
                        elif bndbox is not None:
                            xmin = float(bndbox.find('xmin').text)
                            ymin = float(bndbox.find('ymin').text)
                            xmax = float(bndbox.find('xmax').text)
                            ymax = float(bndbox.find('ymax').text)
                            
                            coords = [
                                xmin / IMG_WIDTH, ymin / IMG_HEIGHT,  # top-left
                                xmax / IMG_WIDTH, ymin / IMG_HEIGHT,  # top-right
                                xmax / IMG_WIDTH, ymax / IMG_HEIGHT,  # bottom-right
                                xmin / IMG_WIDTH, ymax / IMG_HEIGHT,  # bottom-left
                            ]                        
                        if coords:
                            image_to_classes[image_file].add(class_id)
                            label_line = f"{class_id} " + " ".join(map(str, coords)) + "\n"
                            label_file.write(label_line)
                        else:
                            print(f"\nWarning: Object '{class_name}' in XML for '{image_file}' has no valid bounding box. Skipping.")
    return image_to_classes


def prepare_full_dataset():
    print("--- Starting Full Dataset Preparation for Few-Shot YOLO ---")
    if os.path.exists(FEW_SHOT_OUTPUT_DIR):
        print(f"Output directory '{FEW_SHOT_OUTPUT_DIR}' already exists. Deleting it to start fresh.")
        shutil.rmtree(FEW_SHOT_OUTPUT_DIR)

    temp_dir = os.path.join(FEW_SHOT_OUTPUT_DIR, 'temp')
    temp_labels_dir = os.path.join(temp_dir, 'labels')
    sets_to_create = ['base_train', 'finetune_combined', 'val']
    for s in sets_to_create:
        os.makedirs(os.path.join(FEW_SHOT_OUTPUT_DIR, s, 'images'), exist_ok=True)
        os.makedirs(os.path.join(FEW_SHOT_OUTPUT_DIR, s, 'labels'), exist_ok=True)
        
    random.shuffle(ALL_IMAGE_FILES)
    image_to_classes = process_and_convert_labels(ALL_IMAGE_FILES, IMAGE_DIR_TRAINVAL, ANNOTATION_DIR_TRAINVAL, temp_labels_dir, CLASS_MAPPING)

    train_files = ALL_IMAGE_FILES[:TRAIN_SET_SIZE]
    val_files = ALL_IMAGE_FILES[TRAIN_SET_SIZE:]
    print(f"Split dataset: {len(train_files)} train images, {len(val_files)} val images.")

    print("Processing validation set...")
    for image_file in tqdm(val_files, desc="Copying validation files"):
        label_file_path = os.path.join(temp_labels_dir, image_file.replace('.jpg', '.txt'))
        if os.path.exists(label_file_path) and os.path.getsize(label_file_path) > 0:
            shutil.copy(os.path.join(IMAGE_DIR_TRAINVAL, image_file), os.path.join(FEW_SHOT_OUTPUT_DIR, 'val', 'images', image_file))
            shutil.copy(label_file_path, os.path.join(FEW_SHOT_OUTPUT_DIR, 'val', 'labels', image_file.replace('.jpg', '.txt')))

    images_with_base = set()
    images_with_novel = set()
    images_with_only_base = set()
    novel_image_map = defaultdict(list)

    for image_file in tqdm(train_files, desc="Classifying train images"):
        classes_in_image = image_to_classes.get(image_file, set())
        has_base = any(cid in BASE_CLASS_IDS for cid in classes_in_image)
        has_novel = any(cid in NOVEL_CLASS_IDS for cid in classes_in_image)

        if has_base:
            images_with_base.add(image_file)
        if has_novel:
            images_with_novel.add(image_file)
            for cid in classes_in_image:
                if cid in NOVEL_CLASS_IDS:
                    novel_image_map[cid].append(image_file)
        if has_base and not has_novel:
            images_with_only_base.add(image_file)
    
    print(f"Found {len(images_with_base)} images containing base classes.")
    print(f"Found {len(images_with_novel)} images containing novel classes.")
    print(f"Found {len(images_with_only_base)} images containing ONLY base classes.")

    print(f"Creating 'base_train' set...")
    for image_file in tqdm(images_with_base, desc="Creating base_train set"):
        src_label_path = os.path.join(temp_labels_dir, image_file.replace('.jpg', '.txt'))
        if not os.path.exists(src_label_path): continue

        with open(src_label_path, 'r') as f_in:
            lines = f_in.readlines()
        
        base_lines = [line for line in lines if int(line.split()[0]) in BASE_CLASS_IDS]
        
        if base_lines:
            shutil.copy(os.path.join(IMAGE_DIR_TRAINVAL, image_file), os.path.join(FEW_SHOT_OUTPUT_DIR, 'base_train', 'images', image_file))
            dest_label_path = os.path.join(FEW_SHOT_OUTPUT_DIR, 'base_train', 'labels', image_file.replace('.jpg', '.txt'))
            with open(dest_label_path, 'w') as f_out:
                f_out.writelines(base_lines)

    print(f"Creating '{K_SHOT}-shot finetune_combined' set...")
    finetune_images = set()

    for class_id in NOVEL_CLASS_IDS:
        files = list(set(novel_image_map.get(class_id, [])))
        if len(files) > 0:
            selected = random.sample(files, min(K_SHOT, len(files)))
            finetune_images.update(selected)
    
    num_base_samples = len(finetune_images)
    if len(images_with_only_base) > num_base_samples:
        base_samples_for_ft = random.sample(list(images_with_only_base), num_base_samples)
        finetune_images.update(base_samples_for_ft)

    print(f"Total images in finetune set: {len(finetune_images)}")
    for image_file in tqdm(finetune_images, desc="Copying finetune_combined files"):
        src_label_path = os.path.join(temp_labels_dir, image_file.replace('.jpg', '.txt'))
        if os.path.exists(src_label_path) and os.path.getsize(src_label_path) > 0:
            shutil.copy(os.path.join(IMAGE_DIR_TRAINVAL, image_file), os.path.join(FEW_SHOT_OUTPUT_DIR, 'finetune_combined', 'images', image_file))
            shutil.copy(src_label_path, os.path.join(FEW_SHOT_OUTPUT_DIR, 'finetune_combined', 'labels', image_file.replace('.jpg', '.txt')))

    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    print("--- Data preparation complete. ---")


def create_yaml_files():
    print("--- Creating YAML configuration files... ---")
    data_base = {'path': os.path.abspath(FEW_SHOT_OUTPUT_DIR), 'train': 'base_train/images', 'val': 'val/images', 'nc': len(ALL_CLASSES), 'names': ALL_CLASSES}
    with open(os.path.join(FEW_SHOT_OUTPUT_DIR, 'data_base.yaml'), 'w') as f:
        yaml.dump(data_base, f, sort_keys=False)
    data_finetune = {'path': os.path.abspath(FEW_SHOT_OUTPUT_DIR), 'train': 'finetune_combined/images', 'val': 'val/images', 'nc': len(ALL_CLASSES), 'names': ALL_CLASSES}
    with open(os.path.join(FEW_SHOT_OUTPUT_DIR, 'data_finetune.yaml'), 'w') as f:
        yaml.dump(data_finetune, f, sort_keys=False)
    print("--- YAML files created successfully. ---")


if __name__ == '__main__':
    prepare_full_dataset()
    create_yaml_files()