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


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)

    if isinstance(ground_truth_masks, list):
        target_shape = anomaly_segmentations.shape[1:]  # (H, W)
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
        ground_truth_masks = np.stack(resized_masks)
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