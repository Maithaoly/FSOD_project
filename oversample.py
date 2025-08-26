import os
import glob
import shutil

src_images = 'dior_fewshot_data_v2/finetune_combined/images'
src_labels = 'dior_fewshot_data_v2/finetune_combined/labels'

dst_images = 'dior_fewshot_data_v2/finetune_combined_oversample/images'
dst_labels = 'dior_fewshot_data_v2/finetune_combined_oversample/labels'

os.makedirs(dst_images, exist_ok=True)
os.makedirs(dst_labels, exist_ok=True)

repeat_times = 5

image_files = glob.glob(os.path.join(src_images, '*.jpg'))

for i in range(repeat_times):
    for img_path in image_files:
        base_name = os.path.basename(img_path)
        new_name = f"{i}_{base_name}"  # đổi tên để tránh trùng
        shutil.copy(img_path, os.path.join(dst_images, new_name))

        label_path = os.path.join(src_labels, os.path.splitext(base_name)[0] + '.txt')
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(dst_labels, new_name.replace('.jpg', '.txt')))
