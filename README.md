# PatchCore Customization Notes (Kvasir)

This project adapts PatchCore to run on a custom Kvasir-based anomaly-detection dataset arranged in an MVTec-like structure.

## What Was Changed

### 1. Custom dataset preparation pipeline

Added `prepare_dataset.py` to build a PatchCore-compatible folder layout from:

- `kvasir-dataset-v2` (healthy images)
- `Kvasir-SEG` (polyp images + masks)

Configured paths in `prepare_dataset.py`:

- `KVASIR_V2_DIR = ../dataset/kvasir-dataset-v2`
- `KVASIR_SEG_DIR = ../dataset/Kvasir-SEG`
- `OUTPUT_DIR = ../dataset/kvasir_patchcore`

Healthy classes used:

- `normal-cecum`
- `normal-pylorus`
- `normal-z-line`

Train/test split:

- `TRAIN_SPLIT = 0.80`

Generated output tree:

```text
kvasir_patchcore/
	train/
		good/
	test/
		good/
		defect/
	ground_truth/
		defect/
```

Behavior:

- Healthy images from Kvasir-v2 are shuffled with a fixed seed (`42`) and split into `train/good` and `test/good`.
- Polyp images from Kvasir-SEG are copied to `test/defect`.
- Matching segmentation masks are copied to `ground_truth/defect`.

### 2. Kvasir dataset loader updates

Updated `patchcore-inspection/src/patchcore/datasets/kvasir.py`:

- Default class list changed from MVTec classes to:
  - `_CLASSNAMES = ["kvasir_patchcore"]`
- Dataset docstring text updated from MVTec wording to Kvasir wording.
- Added:
  - `self.transform_mean = IMAGENET_MEAN`
  - `self.transform_std = IMAGENET_STD`

Purpose:

- Ensure the loader defaults to the generated Kvasir class folder.
- Keep normalization statistics available as dataset attributes.

## Expected Dataset Location

After running preparation, the training/evaluation scripts should point to:

- `../dataset/kvasir_patchcore`

Inside that path, PatchCore expects one class folder:

- `kvasir_patchcore/`
  - containing `train`, `test`, and `ground_truth` as shown above.

## Quick Run Order

1. Prepare data:

```bash
python prepare_dataset.py
```

2. Train/evaluate PatchCore using the Kvasir dataset path:

- dataset root: `../dataset/kvasir_patchcore`
- class name: `kvasir_patchcore`

## Notes

- This setup treats Kvasir as a one-class anomaly-detection benchmark in PatchCore format.
- `good` = healthy endoscopy images; `defect` = polyp samples with masks.
