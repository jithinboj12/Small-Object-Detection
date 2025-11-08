import os
import shutil
from sklearn.model_selection import train_test_split
import config  # <-- THIS LINE IS NOW FIXED
import yaml

def create_final_dataset_structure():
    print("Creating final train/val splits...")
    
    # --- 1. Get all processed tile names ---
    all_image_files = sorted([f for f in os.listdir(config.PROCESSED_IMAGE_DIR) 
                              if f.endswith('.png')])
    
    # We get the 'basename' so we can find both the image and its label
    basenames = [os.path.splitext(f)[0] for f in all_image_files]
    
    # --- 2. Split the data ---
    train_names, val_names = train_test_split(
        basenames,
        test_size=config.VALIDATION_SPLIT_PERCENT,
        random_state=config.RANDOM_SEED
    )
    
    print(f"Total tiles: {len(basenames)}")
    print(f"Training tiles: {len(train_names)}")
    print(f"Validation tiles: {len(val_names)}")
    
    # --- 3. Create directories and copy files ---
    # Define paths
    train_img_dir = os.path.join(config.FINAL_DATASET_DIR, 'train', 'images')
    train_lbl_dir = os.path.join(config.FINAL_DATASET_DIR, 'train', 'labels')
    val_img_dir = os.path.join(config.FINAL_DATASET_DIR, 'val', 'images')
    val_lbl_dir = os.path.join(config.FINAL_DATASET_DIR, 'val', 'labels')
    
    # Create dirs
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)
    
    # --- 4. Copy files (helper function) ---
    def copy_files(filenames, split_name):
        img_dest_dir = os.path.join(config.FINAL_DATASET_DIR, split_name, 'images')
        lbl_dest_dir = os.path.join(config.FINAL_DATASET_DIR, split_name, 'labels')
        
        for name in filenames:
            # Copy image
            shutil.copy(
                os.path.join(config.PROCESSED_IMAGE_DIR, f"{name}.png"),
                os.path.join(img_dest_dir, f"{name}.png")
            )
            # Copy label
            shutil.copy(
                os.path.join(config.PROCESSED_LABEL_DIR, f"{name}.txt"),
                os.path.join(lbl_dest_dir, f"{name}.txt")
            )
            
    print("Copying training files...")
    copy_files(train_names, 'train')
    
    print("Copying validation files...")
    copy_files(val_names, 'val')
    
    # --- 5. Create data.yaml file ---
    # This file is what the YOLO model (Gopica's task) will read.
    yaml_path = os.path.join(config.FINAL_DATASET_DIR, 'data.yaml')
    
    data_yaml = {
        'train': os.path.abspath(train_img_dir),
        'val': os.path.abspath(val_img_dir),
        'nc': len(config.CLASSES),
        'names': config.CLASSES
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
        
    print(f"\n--- Split Complete ---")
    print(f"Final dataset created at: {config.FINAL_DATASET_DIR}")
    print(f"Dataset config file created at: {yaml_path}")

if __name__ == "__main__":
    create_final_dataset_structure()