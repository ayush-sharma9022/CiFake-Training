"""
Fast Pre-Processing & Pre-Caching Pipeline.
Standardizes images to 128x128 uint8 numpy arrays stored in datasets/processed/images/
Eliminates disk I/O bottlenecks, enabling GPU-accelerated training in seconds.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METADATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "metadata")
PROCESSED_IMG_DIR = os.path.join(PROJECT_ROOT, "datasets", "processed", "images")
os.makedirs(PROCESSED_IMG_DIR, exist_ok=True)


def cache_split(csv_path, out_prefix, target_size=(128, 128)):
    df = pd.read_csv(csv_path)
    print(f"Caching {os.path.basename(csv_path)}: {len(df)} samples...")
    images = []
    labels = df["label"].values.astype(np.float32)

    for p in tqdm(df["path"].values, total=len(df)):
        with Image.open(p) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            if img.size != target_size:
                img = img.resize(target_size, Image.BILINEAR)
            arr = np.array(img, dtype=np.uint8)
            images.append(arr)

    images = np.stack(images, axis=0)
    img_out = os.path.join(PROCESSED_IMG_DIR, f"{out_prefix}_images.npy")
    lbl_out = os.path.join(PROCESSED_IMG_DIR, f"{out_prefix}_labels.npy")

    np.save(img_out, images)
    np.save(lbl_out, labels)
    print(f"Saved {img_out} ({images.shape}, {images.nbytes / (1024*1024):.1f} MB)")


if __name__ == "__main__":
    cache_split(os.path.join(METADATA_DIR, "train_split.csv"), "train")
    cache_split(os.path.join(METADATA_DIR, "val_split.csv"), "val")
    cache_split(os.path.join(METADATA_DIR, "test_split.csv"), "test")
    cache_split(os.path.join(METADATA_DIR, "cross_dataset_test.csv"), "cross")
    print("\nDataset pre-caching complete!")
