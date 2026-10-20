import numpy as np
from scipy import ndimage
from sklearn import metrics
from PIL import Image

# np.trapz was renamed to np.trapezoid in NumPy 2.0; support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": thresholds}


def _prepare_masks(ground_truth_masks, target_shape):
    if isinstance(ground_truth_masks, list):
        resized_masks = []
        for mask in ground_truth_masks:
            mask = np.array(mask)
            mask = np.squeeze(mask)

            # If still not 2D, take first channel
            if mask.ndim == 3:
                mask = mask[0]
            elif mask.ndim == 1:
                mask = mask.reshape(target_shape)

            if mask.shape != target_shape:
                pil_mask = Image.fromarray(mask.astype(np.uint8))
                pil_mask = pil_mask.resize(
                    (target_shape[1], target_shape[0]), Image.NEAREST
                )
                mask = np.array(pil_mask)

            resized_masks.append(mask)
        return np.stack(resized_masks)
    return np.squeeze(np.array(ground_truth_masks))


def compute_binary_segmentation_metrics(binary_maps, ground_truth_masks):
    """Pixel-level Precision/Recall/F1 for binary anomaly segmentation maps.

    Intended to be called on anomalous samples only (the paper's protocol).

    Args:
        binary_maps: list or array of [H, W] maps with values in {0, 1}.
        ground_truth_masks: matching ground-truth masks (any resolution).
    """
    if isinstance(binary_maps, list):
        binary_maps = np.stack(binary_maps)
    binary_maps = (binary_maps > 0).astype(int)

    ground_truth_masks = _prepare_masks(ground_truth_masks, binary_maps.shape[1:])

    preds = binary_maps.ravel()
    gts = (ground_truth_masks.ravel() > 0).astype(int)

    precision = metrics.precision_score(gts, preds, zero_division=0)
    recall = metrics.recall_score(gts, preds, zero_division=0)
    f1 = metrics.f1_score(gts, preds, zero_division=0)
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)

    if isinstance(ground_truth_masks, list):
        ground_truth_masks = _prepare_masks(
            ground_truth_masks, anomaly_segmentations.shape[1:]
        )
    else:
        ground_truth_masks = np.squeeze(np.array(ground_truth_masks))

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }


def compute_pro(
    anomaly_segmentations,
    ground_truth_masks,
    max_fpr: float = 0.3,
    num_thresholds: int = 200,
):
    """Per-Region Overlap (PRO) score.

    For each threshold, every connected component of the ground-truth masks
    contributes its own recall (overlap fraction); the PRO value at that
    threshold is the mean over all regions (so large and small regions count
    equally). PRO is integrated against the false-positive rate over normal
    pixels on ``[0, max_fpr]`` and normalized by ``max_fpr`` so the result lies
    in ``[0, 1]``. This is the standard MVTec-AD / anomaly-segmentation PRO.

    Args:
        anomaly_segmentations: list/array of [H, W] real-valued score maps.
        ground_truth_masks: matching binary masks (any resolution / channels).
        max_fpr: integration limit on the false-positive-rate axis.
        num_thresholds: number of score thresholds swept.

    Returns:
        {"pro": float, "fpr": np.ndarray, "pro_curve": np.ndarray,
         "thresholds": np.ndarray} where the curves are restricted to the
        integrated ``[0, max_fpr]`` region.
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    anomaly_segmentations = np.asarray(anomaly_segmentations, dtype=np.float64)

    ground_truth_masks = _prepare_masks(
        ground_truth_masks, anomaly_segmentations.shape[1:]
    )
    ground_truth_masks = (ground_truth_masks > 0).astype(np.uint8)
    if ground_truth_masks.ndim == 2:
        ground_truth_masks = ground_truth_masks[None]

    # Label connected components per image once (4-connectivity, MVTec default).
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    regions = []  # (image_index, boolean mask, pixel count) per region
    for img_idx, mask in enumerate(ground_truth_masks):
        labeled, n = ndimage.label(mask, structure=structure)
        for region_id in range(1, n + 1):
            region = labeled == region_id
            regions.append((img_idx, region, int(region.sum())))

    normal_pixels = ground_truth_masks == 0
    n_normal = int(normal_pixels.sum())

    if not regions or n_normal == 0:
        # No anomalous regions (or no normal pixels): PRO is undefined; return 0.
        return {
            "pro": 0.0,
            "fpr": np.array([0.0]),
            "pro_curve": np.array([0.0]),
            "thresholds": np.array([0.0]),
        }

    lo = float(anomaly_segmentations.min())
    hi = float(anomaly_segmentations.max())
    thresholds = np.linspace(hi, lo, num_thresholds)

    pros = np.empty(num_thresholds, dtype=np.float64)
    fprs = np.empty(num_thresholds, dtype=np.float64)
    for t_idx, thr in enumerate(thresholds):
        predicted = anomaly_segmentations >= thr
        region_recalls = np.empty(len(regions), dtype=np.float64)
        for r_idx, (img_idx, region, area) in enumerate(regions):
            region_recalls[r_idx] = np.count_nonzero(
                predicted[img_idx][region]
            ) / area
        pros[t_idx] = region_recalls.mean()
        fprs[t_idx] = np.count_nonzero(predicted & normal_pixels) / n_normal

    # Build a proper PRO-vs-FPR curve: sort by FPR and collapse duplicate FPR
    # values to their best (max) PRO, then interpolate onto a dense grid over
    # [0, max_fpr] and integrate. Interpolation is what makes a near-perfect
    # detector (whose thresholds cluster at FPR=0) score ~1 instead of 0.
    order = np.argsort(fprs, kind="stable")
    fpr_sorted = fprs[order]
    pro_sorted = pros[order]
    uniq_fpr, first_idx = np.unique(fpr_sorted, return_index=True)
    uniq_pro = np.maximum.reduceat(pro_sorted, first_idx)
    grid = np.linspace(0.0, max_fpr, 256)
    pro_grid = np.interp(grid, uniq_fpr, uniq_pro, left=uniq_pro[0], right=uniq_pro[-1])
    pro_auc = _trapz(pro_grid, grid) / max_fpr
    return {
        "pro": float(pro_auc),
        "fpr": grid,
        "pro_curve": pro_grid,
        "thresholds": thresholds,
    }

# ---------------------------------------------------------------------------
# Fast exact metrics for the streaming benchmark. A Kvasir run scores ~80M
# pixels about 350 times; the reference implementations above sort all
# pixels three times (pixelwise) and loop 200 thresholds x every region in
# Python (PRO), which made evaluation dominate the benchmark's wall clock.
# Both functions below return the SAME numbers (tested against the originals).
# ---------------------------------------------------------------------------
def pixel_auroc(anomaly_segmentations, ground_truth_masks, device=None):
    """Exact pixel AUROC (== sklearn roc_auc_score) from one sort.

    Mann-Whitney U with tie-averaged ranks, which is precisely what
    roc_auc_score computes. Runs on ``device`` (default: CUDA when available).
    """
    import torch

    segs = np.asarray(anomaly_segmentations, dtype=np.float32).ravel()
    gt = _prepare_masks(list(ground_truth_masks), np.asarray(anomaly_segmentations).shape[1:]) \
        if isinstance(ground_truth_masks, list) else np.asarray(ground_truth_masks)
    pos = (np.squeeze(gt).ravel() > 0)
    n_pos = int(pos.sum())
    n_neg = pos.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    dev = torch.device(device) if device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    s = torch.from_numpy(segs).to(dev)
    p = torch.from_numpy(pos).to(dev)
    uniq, inverse, counts = torch.unique(s, sorted=True, return_inverse=True,
                                         return_counts=True)
    # average 1-based rank of each distinct value
    upper = torch.cumsum(counts.to(torch.float64), 0)
    avg_rank = upper - (counts.to(torch.float64) - 1.0) / 2.0
    r_pos = avg_rank[inverse[p]].sum().item()
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compute_pro_fast(anomaly_segmentations, ground_truth_masks,
                     max_fpr: float = 0.3, num_thresholds: int = 200):
    """``compute_pro`` without the Python threshold x region loop.

    Each pixel's score is mapped once to the number of thresholds it clears
    (searchsorted on the same float64 threshold grid, so ``score >= thr`` is
    decided identically); per-region and normal-pixel histograms of that
    index, cumulated, give every threshold's region recalls and FPR at once.
    Same thresholds, same curve post-processing, same result.
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    segs = np.asarray(anomaly_segmentations, dtype=np.float64)
    gts = _prepare_masks(ground_truth_masks, segs.shape[1:])
    gts = (gts > 0).astype(np.uint8)
    if gts.ndim == 2:
        gts = gts[None]

    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    labels = np.zeros(gts.shape, dtype=np.int64)
    n_regions = 0
    for i, mask in enumerate(gts):
        lab, n = ndimage.label(mask, structure=structure)
        labels[i] = np.where(lab > 0, lab + n_regions, 0)
        n_regions += n
    normal = gts == 0
    n_normal = int(normal.sum())
    if n_regions == 0 or n_normal == 0:
        return {"pro": 0.0, "fpr": np.array([0.0]), "pro_curve": np.array([0.0]),
                "thresholds": np.array([0.0])}

    lo, hi = float(segs.min()), float(segs.max())
    thresholds = np.linspace(hi, lo, num_thresholds)       # descending, as before
    asc = thresholds[::-1]
    T = num_thresholds
    # k = number of thresholds <= score  =>  pixel predicted at asc[i] iff i < k
    k = np.searchsorted(asc, segs.ravel(), side="right")
    lab = labels.ravel()
    in_region = lab > 0
    hist = np.bincount((lab[in_region] - 1) * (T + 1) + k[in_region],
                       minlength=n_regions * (T + 1)).reshape(n_regions, T + 1)
    area = hist.sum(axis=1).astype(np.float64)
    ge = np.cumsum(hist[:, ::-1], axis=1)[:, ::-1]         # ge[:, m] = #pixels with k >= m
    recalls_asc = ge[:, 1:] / area[:, None]                  # asc index i needs k >= i+1
    pros = recalls_asc.mean(axis=0)[::-1]                    # back to descending order
    nh = np.bincount(k[normal.ravel()], minlength=T + 1)
    nge = np.cumsum(nh[::-1])[::-1]
    fprs = (nge[1:] / n_normal)[::-1]

    order = np.argsort(fprs, kind="stable")
    fpr_sorted, pro_sorted = fprs[order], pros[order]
    uniq_fpr, first_idx = np.unique(fpr_sorted, return_index=True)
    uniq_pro = np.maximum.reduceat(pro_sorted, first_idx)
    grid = np.linspace(0.0, max_fpr, 256)
    pro_grid = np.interp(grid, uniq_fpr, uniq_pro, left=uniq_pro[0], right=uniq_pro[-1])
    return {"pro": float(_trapz(pro_grid, grid) / max_fpr), "fpr": grid,
            "pro_curve": pro_grid, "thresholds": thresholds}
