"""
prepare_dataset_capsule.py
==========================
Converts the Kvasir-Capsule dataset (raw videos + metadata.csv) into an
MVTec-AD-compatible folder layout so that PatchCore can load it directly
via its existing MVTecDataset class.

Target layout
─────────────
  <output_root>/
  └── capsule/
      ├── train/
      │   └── good/          ← normal frames (train split)
      ├── test/
      │   ├── good/          ← normal frames (test split)
      │   └── anomaly/       ← all anomaly frames
      └── ground_truth/
          └── anomaly/       ← black masks (same filenames as test/anomaly/)

Usage
─────
  python prepare_dataset_capsule.py \
      --labelled_video_dir  /path/to/labelled_videos \
      --unlabelled_video_dir /path/to/unlabelled_videos \
      --metadata_csv        /path/to/metadata.csv \
      --output_root         /path/to/output/kvasir_capsule_patchcore

After running, train PatchCore with:
  dataset --resize 256 --imagesize 224 -d capsule mvtec <output_root>
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ─── Default class definitions ─────────────────────────────────────────────
NORMAL_CLASS = "Normal clean mucosa"
ANOMALY_CLASSES = [
    "Polyp", "Bleeding", "Ulcer", "Foreign Body",
    "Erosion", "Erythema", "Angiectasia", "Lymphangiectasia",
]


# ─── Argument parsing ──────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Kvasir-Capsule videos → MVTec-format folders for PatchCore."
    )
    parser.add_argument(
        "--labelled_video_dir",
        type=str,
        default="/home/user1/aniket/Patchcore/dataset/data/labelled_videos",
        help="Directory containing labelled .mp4/.avi videos.",
    )
    parser.add_argument(
        "--unlabelled_video_dir",
        type=str,
        default="/home/user1/aniket/Patchcore/dataset/data/kvasir-capsule-unlabeled-videos",
        help="Directory containing unlabelled .mp4/.avi videos.",
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default="/home/user1/aniket/Patchcore/dataset/data/metadata.csv",
        help="Path to metadata.csv (semicolon-delimited).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/home/user1/aniket/Patchcore/dataset/kvasir_capsule_patchcore",
        help="Root output directory. A 'capsule/' subfolder is created inside.",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Fraction of normal videos used for training (rest → test/good).",
    )
    parser.add_argument(
        "--unlabelled_every_n",
        type=int,
        default=30,
        help="Sample every N-th frame from unlabelled videos (default: 30).",
    )
    parser.add_argument(
        "--max_unlabelled_frames",
        type=int,
        default=5000,
        help="Maximum total frames to extract from unlabelled videos (0 = unlimited).",
    )
    parser.add_argument(
        "--skip_unlabelled",
        action="store_true",
        help="Skip unlabelled videos entirely.",
    )
    return parser.parse_args()


# ─── Metadata parser ───────────────────────────────────────────────────────
def parse_metadata(csv_path):
    """
    Parse the Kvasir-Capsule metadata.csv.

    Returns
    -------
    normal_frames  : dict  {video_id: [frame_number, ...]}
    anomaly_frames : dict  {video_id: [(frame_number, class_name), ...]}
    """
    normal_frames = defaultdict(list)
    anomaly_frames = defaultdict(list)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            vid = row["video_id"].strip()
            fnum = int(row["frame_number"].strip())
            cls = row["finding_class"].strip()

            if cls == NORMAL_CLASS:
                normal_frames[vid].append(fnum)
            elif cls in ANOMALY_CLASSES:
                anomaly_frames[vid].append((fnum, cls))

    return normal_frames, anomaly_frames


# ─── Frame extraction ──────────────────────────────────────────────────────
def extract_specific_frames(video_path, frame_numbers, output_dir, filename_prefix):
    """
    Extract specific frame numbers from a video and save as PNG.

    Filenames are formatted as: {filename_prefix}_frame_{frame_idx:06d}.png
    This prevents collisions when multiple videos write to the same folder.

    Parameters
    ----------
    video_path      : str   Path to video file.
    frame_numbers   : list  Frame indices to extract.
    output_dir      : str   Directory to write PNGs to.
    filename_prefix : str   Prefix for output filenames (typically the video_id).

    Returns
    -------
    saved : int   Number of frames successfully written.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Could not open video {video_path}")
        return 0

    frame_set = set(frame_numbers)
    max_needed = max(frame_set) if frame_set else -1
    saved = 0
    frame_idx = 0

    while frame_idx <= max_needed:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in frame_set:
            out_path = os.path.join(
                output_dir, f"{filename_prefix}_frame_{frame_idx:06d}.png"
            )
            cv2.imwrite(out_path, frame)
            saved += 1
        frame_idx += 1

    cap.release()
    return saved


def find_video(video_id, *dirs):
    """Search for a video file (mp4/avi) across multiple directories."""
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for ext in [".mp4", ".avi"]:
            p = os.path.join(d, video_id + ext)
            if os.path.exists(p):
                return p
    return None


# ─── Black mask creator ────────────────────────────────────────────────────
def create_black_masks(image_dir, mask_dir):
    """
    For every PNG in image_dir, create a same-sized all-black mask in mask_dir.
    This satisfies PatchCore's requirement for ground_truth masks without
    having pixel-level annotations (image-level AUROC will still work).
    """
    os.makedirs(mask_dir, exist_ok=True)
    count = 0
    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".bmp", ".tif")):
            continue
        img = cv2.imread(os.path.join(image_dir, fname))
        if img is None:
            continue
        h, w = img.shape[:2]
        black_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.imwrite(os.path.join(mask_dir, fname), black_mask)
        count += 1
    return count


# ─── Main build pipeline ───────────────────────────────────────────────────
def build_dataset(args):
    # ── Define output paths (MVTec layout) ──
    base = Path(args.output_root) / "capsule"
    train_good = base / "train" / "good"
    test_good = base / "test" / "good"
    test_anomaly = base / "test" / "anomaly"
    gt_anomaly = base / "ground_truth" / "anomaly"

    for d in [train_good, test_good, test_anomaly, gt_anomaly]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Parse metadata ──
    print("=" * 60)
    print("STEP 1: Parsing metadata")
    print("=" * 60)
    normal_frames, anomaly_frames = parse_metadata(args.metadata_csv)
    print(f"  Normal videos  : {len(normal_frames)}")
    print(f"  Anomaly videos : {len(anomaly_frames)}")
    total_normal = sum(len(v) for v in normal_frames.values())
    total_anomaly = sum(len(v) for v in anomaly_frames.values())
    print(f"  Normal frames  : {total_normal}")
    print(f"  Anomaly frames : {total_anomaly}")

    # ── Step 2: Extract normal frames → train/good + test/good ──
    print()
    print("=" * 60)
    print("STEP 2: Extracting normal frames (train/good + test/good)")
    print("=" * 60)
    all_normal_vids = sorted(normal_frames.keys())
    split_idx = int(len(all_normal_vids) * args.train_split)

    train_count = 0
    test_good_count = 0
    for i, vid in enumerate(all_normal_vids):
        vpath = find_video(vid, args.labelled_video_dir, args.unlabelled_video_dir)
        if vpath is None:
            print(f"  WARNING: Video '{vid}' not found, skipping")
            continue
        frames = normal_frames[vid]
        if i < split_idx:
            n = extract_specific_frames(vpath, frames, str(train_good), vid)
            train_count += n
            print(f"  {vid} → train/good ({n} frames)")
        else:
            n = extract_specific_frames(vpath, frames, str(test_good), vid)
            test_good_count += n
            print(f"  {vid} → test/good  ({n} frames)")

    # ── Step 3: Extract anomaly frames → test/anomaly ──
    print()
    print("=" * 60)
    print("STEP 3: Extracting anomaly frames (test/anomaly)")
    print("=" * 60)
    anomaly_count = 0
    for vid, frame_list in sorted(anomaly_frames.items()):
        vpath = find_video(vid, args.labelled_video_dir, args.unlabelled_video_dir)
        if vpath is None:
            print(f"  WARNING: Video '{vid}' not found, skipping")
            continue
        frames = [fnum for fnum, cls in frame_list]
        n = extract_specific_frames(vpath, frames, str(test_anomaly), vid)
        anomaly_count += n
        print(f"  {vid} → test/anomaly ({n} frames)")

    # ── Step 4: Create black masks for anomaly frames ──
    print()
    print("=" * 60)
    print("STEP 4: Creating black masks (ground_truth/anomaly)")
    print("=" * 60)
    mask_count = create_black_masks(str(test_anomaly), str(gt_anomaly))
    print(f"  Created {mask_count} black masks")

    # ── Step 5 (optional): Extract from unlabelled videos → train/good ──
    unlabelled_count = 0
    if not args.skip_unlabelled and os.path.isdir(args.unlabelled_video_dir):
        print()
        print("=" * 60)
        print("STEP 5: Sampling unlabelled videos (train/good)")
        print("=" * 60)
        total_unlabelled_so_far = 0
        for vfile in sorted(Path(args.unlabelled_video_dir).glob("*.mp4")):
            vid = vfile.stem
            # Skip videos already processed as labelled
            if vid in normal_frames or vid in anomaly_frames:
                continue
            # Check cap
            if (
                args.max_unlabelled_frames > 0
                and total_unlabelled_so_far >= args.max_unlabelled_frames
            ):
                print(f"  Reached max unlabelled frame cap ({args.max_unlabelled_frames})")
                break

            cap = cv2.VideoCapture(str(vfile))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            frames = list(range(0, total_frames, args.unlabelled_every_n))
            # Respect the cap
            if args.max_unlabelled_frames > 0:
                remaining = args.max_unlabelled_frames - total_unlabelled_so_far
                frames = frames[:remaining]

            n = extract_specific_frames(str(vfile), frames, str(train_good), vid)
            unlabelled_count += n
            total_unlabelled_so_far += n
            print(f"  {vid} → train/good ({n} frames)")
    else:
        print()
        print("STEP 5: Skipping unlabelled videos")

    # ── Summary ──
    print()
    print("=" * 60)
    print("DONE — Final Summary")
    print("=" * 60)
    print(f"  train/good          : {train_count + unlabelled_count} frames")
    print(f"    ├─ from labelled  : {train_count}")
    print(f"    └─ from unlabelled: {unlabelled_count}")
    print(f"  test/good           : {test_good_count} frames")
    print(f"  test/anomaly        : {anomaly_count} frames")
    print(f"  ground_truth/anomaly: {mask_count} masks")
    print()
    print("Output directory structure:")
    for d in sorted(base.rglob("*")):
        if d.is_dir():
            count = len([f for f in d.iterdir() if f.is_file()])
            if count > 0:
                print(f"  {d.relative_to(base)}: {count} files")
    print()
    print("To train PatchCore, run:")
    print(f'  dataset --resize 256 --imagesize 224 -d capsule mvtec "{args.output_root}"')


if __name__ == "__main__":
    args = parse_args()
    build_dataset(args)