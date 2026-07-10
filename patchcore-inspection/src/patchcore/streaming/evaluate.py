"""Evaluate a memory bank on a (cached) labeled test set.

Scoring goes through the stock :class:`patchcore.common.NearestNeighbourScorer`
so anomaly scores are identical to what stock PatchCore would produce for the
same bank vectors — only the memory-bank *contents* differ across policies.
"""
from typing import Optional, Tuple

import numpy as np
import torch

import patchcore.common
import patchcore.metrics
from patchcore.patchcore import PatchMaker


def _image_and_pixel_scores(
    bank,
    reader,
    n_nearest_neighbours: int,
    patch_shape: Optional[Tuple[int, int]],
    imagesize: Optional[int],
    device: str,
):
    """Return (image_scores [N], segmentations [N,H,W] or None)."""
    scorer = patchcore.common.NearestNeighbourScorer(
        n_nearest_neighbours=min(n_nearest_neighbours, max(len(bank), 1)),
        nn_method=patchcore.common.FaissNN(False, 4),
    )
    bank.install_into(scorer)

    n, p, d = reader.embeddings.shape
    flat = np.ascontiguousarray(reader.embeddings.reshape(n * p, d), dtype=np.float32)
    patch_scores = scorer.predict([flat])[0]  # mean of k NN distances per patch
    patch_scores = patch_scores.reshape(n, p)

    maker = PatchMaker(3, stride=1)
    image_scores = maker.score(patch_scores.copy())  # max over patches

    segmentations = None
    if patch_shape is not None and imagesize is not None and reader.masks is not None:
        h, w = patch_shape
        if h * w == p:
            grid = patch_scores.reshape(n, h, w)
            segmentor = patchcore.common.RescaleSegmentor(
                device=torch.device(device), target_size=(imagesize, imagesize)
            )
            segmentations = np.stack(segmentor.convert_to_segmentation(grid))
    return np.asarray(image_scores, dtype=np.float64), segmentations


def evaluate_bank_on_stage(
    bank,
    reader,
    n_nearest_neighbours: int = 1,
    patch_shape: Optional[Tuple[int, int]] = None,
    imagesize: Optional[int] = None,
    device: str = "cpu",
) -> dict:
    """Compute image AUROC (+ pixel AUROC / PRO when masks are available).

    ``reader`` is an :class:`EmbeddingCacheReader` or synthetic ``ArrayReader``
    with ``embeddings``, ``labels``, and optionally ``masks``.
    """
    labels = np.asarray(reader.labels, dtype=int)
    image_scores, segmentations = _image_and_pixel_scores(
        bank, reader, n_nearest_neighbours, patch_shape, imagesize, device
    )

    out = {"n_images": int(len(labels)), "n_bank": int(len(bank))}
    if len(np.unique(labels)) > 1:
        out["image_auroc"] = float(
            patchcore.metrics.compute_imagewise_retrieval_metrics(
                image_scores, labels
            )["auroc"]
        )
    else:
        out["image_auroc"] = float("nan")

    if segmentations is not None and reader.masks is not None:
        masks = np.asarray(reader.masks)
        out["pixel_auroc"] = float(
            patchcore.metrics.compute_pixelwise_retrieval_metrics(
                list(segmentations), list(masks)
            )["auroc"]
        )
        anom = [i for i in range(len(masks)) if masks[i].sum() > 0]
        if anom:
            out["pro"] = float(
                patchcore.metrics.compute_pro(
                    [segmentations[i] for i in anom], [masks[i] for i in anom]
                )["pro"]
            )
    return out
