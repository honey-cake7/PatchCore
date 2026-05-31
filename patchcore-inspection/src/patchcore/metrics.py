import numpy as np
from sklearn import metrics
from PIL import Image


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