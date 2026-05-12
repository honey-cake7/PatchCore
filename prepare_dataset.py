import os
import shutil
import random
from pathlib import Path

# ==========================================
# 1. SET YOUR PATHS HERE
# ==========================================
# Path to the unzipped Kvasir-v2 dataset folder
KVASIR_V2_DIR = Path("dataset/kvasir-dataset-v2/kvasir-dataset-v2") 

# Path to the unzipped Kvasir-SEG dataset folder
KVASIR_SEG_DIR = Path("dataset/kvasir-seg/Kvasir-SEG")       

# Where you want the new, MVTec-formatted dataset to be saved
OUTPUT_DIR = Path("dataset/custom_kvasir")        

# ==========================================
# 2. CONFIGURATION
# ==========================================
# We ONLY want completely healthy tissue for PatchCore's memory bank
HEALTHY_CLASSES = ["normal-cecum", "normal-pylorus", "normal-z-line"]
TRAIN_SPLIT = 0.80 # 80% of healthy images for training, 20% for testing

def create_structure(base_dir):
    """Creates the exact folder tree required by PatchCore/TTT4AS."""
    paths = {
        "train_good": base_dir / "train" / "good",
        "test_good": base_dir / "test" / "good",
        "test_defect": base_dir / "test" / "defect",
        "gt_defect": base_dir / "ground_truth" / "defect"
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths

def process_healthy_data(kvasir_v2_path, out_paths):
    """Splits healthy classes into train/good and test/good."""
    print("Processing healthy nominal data from Kvasir-v2...")
    
    all_healthy_images = []
    
    # Gather all images from the healthy subfolders
    for healthy_class in HEALTHY_CLASSES:
        class_dir = kvasir_v2_path / healthy_class
        if class_dir.exists():
            images = list(class_dir.glob("*.jpg"))
            all_healthy_images.extend(images)
            print(f"Found {len(images)} images in {healthy_class}")
        else:
            print(f"Warning: Could not find folder {class_dir}")

    # Shuffle to ensure a random distribution of different healthy tissues
    random.seed(42) # Fixed seed for reproducibility
    random.shuffle(all_healthy_images)
    
    # Calculate split index
    split_idx = int(len(all_healthy_images) * TRAIN_SPLIT)
    train_images = all_healthy_images[:split_idx]
    test_images = all_healthy_images[split_idx:]
    
    # Copy files
    for img in train_images:
        shutil.copy(img, out_paths["train_good"] / img.name)
    for img in test_images:
        shutil.copy(img, out_paths["test_good"] / img.name)
        
    print(f"-> Copied {len(train_images)} to train/good")
    print(f"-> Copied {len(test_images)} to test/good")

def process_anomaly_data(kvasir_seg_path, out_paths):
    """Moves polyp images to test/defect and their masks to ground_truth/defect."""
    print("\nProcessing anomalous data from Kvasir-SEG...")
    
    seg_images_dir = kvasir_seg_path / "images"
    seg_masks_dir = kvasir_seg_path / "masks"
    
    if not seg_images_dir.exists() or not seg_masks_dir.exists():
        print("Error: Kvasir-SEG 'images' or 'masks' folder not found. Check your paths.")
        return

    anomaly_images = list(seg_images_dir.glob("*.jpg"))
    
    # Ensure every image has a matching mask before copying
    copied_count = 0
    for img in anomaly_images:
        mask = seg_masks_dir / img.name
        if mask.exists():
            shutil.copy(img, out_paths["test_defect"] / img.name)
            shutil.copy(mask, out_paths["gt_defect"] / mask.name)
            copied_count += 1
            
    print(f"-> Copied {copied_count} images to test/defect")
    print(f"-> Copied {copied_count} masks to ground_truth/defect")

if __name__ == "__main__":
    print("Initializing dataset split...\n")
    out_paths = create_structure(OUTPUT_DIR)
    
    process_healthy_data(KVASIR_V2_DIR, out_paths)
    process_anomaly_data(KVASIR_SEG_DIR, out_paths)
    
    print("\nDataset split complete! You are ready to train PatchCore.")
