from ultralytics import YOLO
import os
import torch
import shutil
import random
from datetime import datetime

# =============================================================================
# --- SETTING ---
# =============================================================================
DATA_DIR = 'dior_fewshot_data_v2'
# BASE_MODEL = 'yolov8n-obb.pt'
BASE_MODEL = 'yolo11n-obb.pt'
PROJECT_NAME = 'DIOR_FewShot_Project'
BASE_TRAIN_EPOCHS = 50
FINETUNE_EPOCHS = 30
IMAGE_SIZE = 800
BATCH_SIZE = 16
FREEZE_LAYERS = 10
FINETUNE_LR0 = 0.001
FINETUNE_LRF = 0.01   

# =============================================================================
# --- MAIN PIPELINE ---
# =============================================================================
def main():
    print("Main function started.")
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- STAGE 1: BASE TRAIN ---
    print("\n" + "="*50 + "\n STAGE 1: BASE TRAINING on base classes\n" + "="*50)
    model_base = YOLO(BASE_MODEL)
    
    model_base.train(
        data=os.path.join(DATA_DIR, 'data_base.yaml'),
        epochs=BASE_TRAIN_EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_NAME,
        name='1_base_training',
        exist_ok=True,
        patience=10
    )
    
    base_weights_path = os.path.join(PROJECT_NAME, '1_base_training', 'weights', 'last.pt')
    print(f"\n--- Base training complete. Best weights at: {base_weights_path} ---")

    # --- STAGE 2: FINETUNE ---
    print("\n" + "="*50 + "\n STAGE 2: FINE-TUNING with frozen backbone\n" + "="*50)
    if not os.path.exists(base_weights_path):
        print(f"ERROR: Base weights not found at {base_weights_path}. Cannot finetune.")
        return
        
    model_finetune = YOLO(base_weights_path)
    

    model_finetune.train(
        data=os.path.join(DATA_DIR, 'data_finetune.yaml'),
        epochs=FINETUNE_EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_NAME,
        name='2_finetuning',
        exist_ok=True,
        freeze=FREEZE_LAYERS,
        lr0=FINETUNE_LR0,
        lrf=FINETUNE_LRF,
        augment=True,
        # patience=5
    )
    final_weights_path = os.path.join(PROJECT_NAME, '2_finetuning', 'weights', 'last.pt')
    print("\n" + "="*50 + "\n FEW-SHOT TRAINING PIPELINE COMPLETE!\n" + f" Final model saved at: {final_weights_path}\n" + "="*50)

if __name__ == '__main__':
    main()