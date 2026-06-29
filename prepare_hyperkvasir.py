"""
prepare_hyperkvasir.py
======================
Converts HyperKvasir dataset into MVTec-AD-compatible format for PatchCore.

Target layout:
  <output_root>/
  └── hyperkvasir/
      ├── train/
      │   └── good/          ← anatomical landmarks (normal) - 80% split
      ├── test/
      │   ├── good/          ← anatomical landmarks (normal) - 20% split
      │   └── anomaly/       ← segmented polyp images
      └── ground_truth/
          └── anomaly/       ← real pixel-level masks
"""

import os
import shutil
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────
HYPERKVASIR_ROOT = '/home/user1/aniket/Patchcore/dataset/hyperkvasir'
OUTPUT_ROOT      = '/home/user1/aniket/Patchcore/dataset/hyperkvasir_patchcore'
TRAIN_SPLIT      = 0.8
# ──────────────────────────────────────────────────────────

# Normal classes — anatomical landmarks only
NORMAL_DIRS = [
    'labeled-images/upper-gi-tract/anatomical-landmarks/pylorus',
    'labeled-images/upper-gi-tract/anatomical-landmarks/retroflex-stomach',
    'labeled-images/upper-gi-tract/anatomical-landmarks/z-line',
    'labeled-images/lower-gi-tract/anatomical-landmarks/retroflex-rectum',
    'labeled-images/lower-gi-tract/anatomical-landmarks/ileum',
    'labeled-images/lower-gi-tract/anatomical-landmarks/cecum',
]

# Segmented images — polyps with real pixel masks
SEGMENTED_IMAGES_DIR = 'segmented-images/images'
SEGMENTED_MASKS_DIR  = 'segmented-images/masks'


def collect_images(directory):
    """Collect all image files from a directory."""
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    images = []
    if not os.path.exists(directory):
        return images
    for f in sorted(os.listdir(directory)):
        if Path(f).suffix.lower() in exts:
            images.append(os.path.join(directory, f))
    return images


def copy_images(src_list, dst_dir, prefix=''):
    """Copy images to destination directory."""
    os.makedirs(dst_dir, exist_ok=True)
    copied = 0
    for src in src_list:
        fname = prefix + os.path.basename(src)
        dst = os.path.join(dst_dir, fname)
        shutil.copy2(src, dst)
        copied += 1
    return copied


def build_dataset():
    base      = Path(OUTPUT_ROOT) / 'hyperkvasir'
    train_good  = base / 'train'        / 'good'
    test_good   = base / 'test'         / 'good'
    test_anomaly = base / 'test'        / 'anomaly'
    gt_anomaly  = base / 'ground_truth' / 'anomaly'

    for d in [train_good, test_good, test_anomaly, gt_anomaly]:
        d.mkdir(parents=True, exist_ok=True)

    root = Path(HYPERKVASIR_ROOT)

    # ── Step 1: Normal images → train/good + test/good ──
    print("=" * 60)
    print("STEP 1: Collecting normal images (anatomical landmarks)")
    print("=" * 60)

    all_normal = []
    for rel_dir in NORMAL_DIRS:
        d = root / rel_dir
        imgs = collect_images(str(d))
        print(f"  {rel_dir}: {len(imgs)} images")
        # Add class prefix to avoid filename collisions
        class_name = rel_dir.split('/')[-1]
        for img in imgs:
            all_normal.append((img, class_name))

    # Split 80/20
    split_idx = int(len(all_normal) * TRAIN_SPLIT)
    train_imgs = all_normal[:split_idx]
    test_imgs  = all_normal[split_idx:]

    print(f"\n  Total normal: {len(all_normal)}")
    print(f"  Train split : {len(train_imgs)}")
    print(f"  Test split  : {len(test_imgs)}")

    # Copy train
    print("\n  Copying to train/good...")
    n = 0
    for img_path, cls in train_imgs:
        fname = f"{cls}_{os.path.basename(img_path)}"
        shutil.copy2(img_path, str(train_good / fname))
        n += 1
    print(f"  Copied {n} images to train/good")

    # Copy test good
    print("  Copying to test/good...")
    n = 0
    for img_path, cls in test_imgs:
        fname = f"{cls}_{os.path.basename(img_path)}"
        shutil.copy2(img_path, str(test_good / fname))
        n += 1
    print(f"  Copied {n} images to test/good")

    # ── Step 2: Segmented images → test/anomaly ──
    print("\n" + "=" * 60)
    print("STEP 2: Copying segmented polyp images → test/anomaly")
    print("=" * 60)

    seg_images_dir = str(root / SEGMENTED_IMAGES_DIR)
    seg_masks_dir  = str(root / SEGMENTED_MASKS_DIR)

    seg_images = collect_images(seg_images_dir)
    seg_masks  = collect_images(seg_masks_dir)

    print(f"  Segmented images: {len(seg_images)}")
    print(f"  Segmented masks : {len(seg_masks)}")

    # Copy images → test/anomaly
    n = copy_images(seg_images, str(test_anomaly))
    print(f"  Copied {n} images to test/anomaly")

    # Copy masks → ground_truth/anomaly (keep exact same filenames!)
    print("\n" + "=" * 60)
    print("STEP 3: Copying real pixel masks → ground_truth/anomaly")
    print("=" * 60)
    n = copy_images(seg_masks, str(gt_anomaly))
    print(f"  Copied {n} masks to ground_truth/anomaly")

    # ── Step 3: Verify mask-image filename matching ──
    print("\n" + "=" * 60)
    print("STEP 4: Verifying mask-image filename matching")
    print("=" * 60)

    anomaly_files = set(os.listdir(str(test_anomaly)))
    mask_files    = set(os.listdir(str(gt_anomaly)))
    matched   = anomaly_files & mask_files
    unmatched = anomaly_files - mask_files

    print(f"  Matched pairs  : {len(matched)}")
    print(f"  Images without masks: {len(unmatched)}")
    if unmatched:
        print("  WARNING - these images have no mask:")
        for f in sorted(unmatched)[:5]:
            print(f"    {f}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("DONE — Final Summary")
    print("=" * 60)
    for d in sorted(base.rglob('*')):
        if d.is_dir():
            count = len([f for f in d.iterdir() if f.is_file()])
            if count > 0:
                print(f"  {d.relative_to(base)}: {count} files")

    print("\nTo train PatchCore run:")
    print(f'  dataset --resize 256 --imagesize 224 -d hyperkvasir mvtec "{OUTPUT_ROOT}"')


if __name__ == '__main__':
    build_dataset()