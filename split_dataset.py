"""
Split dataset into train, val, and test sets for YOLO training.

Usage:
  python split_dataset.py
"""

import os
import shutil
import random
from glob import glob

# -------- USER CONFIG --------
SOURCE_ROOT = "datasets/raw_data"     # folder containing raw data
SOURCE_IMAGES = "images"              # subfolder containing all images
SOURCE_LABELS = "labels"              # subfolder containing all labels
OUTPUT_ROOT = "datasets/toy_car"      # target output folder for YOLO format

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
# -----------------------------

def make_dirs(path):
    """Create train/val/test folder structure."""
    for subset in ["train", "val", "test"]:
        os.makedirs(os.path.join(path, "images", subset), exist_ok=True)
        os.makedirs(os.path.join(path, "labels", subset), exist_ok=True)

def copy_pair(img_path, subset):
    """Copy image and matching label file into subset folder."""
    fname = os.path.basename(img_path)
    stem, _ = os.path.splitext(fname)
    lbl_path = os.path.join(SOURCE_ROOT, SOURCE_LABELS, stem + ".txt")

    img_dest = os.path.join(OUTPUT_ROOT, "images", subset, fname)
    lbl_dest = os.path.join(OUTPUT_ROOT, "labels", subset, stem + ".txt")

    shutil.copy2(img_path, img_dest)
    if os.path.exists(lbl_path):
        shutil.copy2(lbl_path, lbl_dest)
    else:
        print(f"⚠️  Missing label for: {fname}")

def main():
    # Collect image files
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_files.extend(glob(os.path.join(SOURCE_ROOT, SOURCE_IMAGES, ext)))
    image_files.sort()
    random.shuffle(image_files)

    total = len(image_files)
    n_train = int(total * TRAIN_RATIO)
    n_val = int(total * VAL_RATIO)
    n_test = total - n_train - n_val

    print(f"Total images found: {total}")
    print(f"→ Train: {n_train}, Val: {n_val}, Test: {n_test}")

    # Create destination structure
    make_dirs(OUTPUT_ROOT)

    # Split and copy files
    for i, img_path in enumerate(image_files):
        if i < n_train:
            subset = "train"
        elif i < n_train + n_val:
            subset = "val"
        else:
            subset = "test"
        copy_pair(img_path, subset)

    print("\n✅ Dataset split complete!")
    print(f"Results saved in: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()