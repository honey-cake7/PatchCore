"""Evaluate a memory bank on a (cached) labeled test set.

Scoring goes through the stock :class:`patchcore.common.NearestNeighbourScorer`
so anomaly scores are identical to what stock PatchCore would produce for the
same bank vectors — only the memory-bank *contents* differ across policies.
"""
import os
from typing import Optional, Tuple

import numpy as np
import torch

import patchcore.common
import patchcore.metrics
from patchcore.patchcore import PatchMaker


def _available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))  # respects SLURM/cgroup allocation
    except AttributeError:  # non-Linux
        return os.cpu_count() or 4


def _torch_patch_scores(bank, flat: np.ndarray, k: int) -> np.ndarray:
    """Mean of the k smallest *squared* L2 distances to the bank, per patch.

    Numerically matches ``NearestNeighbourScorer.predict`` (FAISS IndexFlatL2
    returns squared L2 and the scorer means over k) up to float noise, but runs
    chunked on the k-NN device — at large M this turns each per-stage eval from
    minutes of CPU FAISS into seconds.
    """
    from patchcore.streaming.bank import knn_device

    base = torch.from_numpy(bank.vectors()).to(knn_device())
    kk = min(k, len(base))
    out = np.empty(len(flat), dtype=np.float32)
    chunk = max(256, 64_000_000 // max(len(base), 1))
    for s in range(0, len(flat), chunk):
        q = torch.from_numpy(flat[s:s + chunk]).to(base.device)
        d = torch.cdist(q, base) ** 2
        vals, _ = torch.topk(d, kk, dim=1, largest=False)
        out[s:s + chunk] = vals.mean(dim=1).cpu().numpy()
    return out


def _image_and_pixel_scores(
    bank,
    reader,
    n_nearest_neighbours: int,
    patch_shape: Optional[Tuple[int, int]],
    imagesize: Optional[int],
    device: str,
):
    """Return (image_scores [N], segmentations [N,H,W] or None).

    Scoring backend: ``STREAMING_EVAL_BACKEND=faiss`` (default; byte-identical
    to stock PatchCore) or ``torch`` (same scores up to float noise, chunked on
    the k-NN device — much faster at large bank capacities).
    """
    n, p, d = reader.embeddings.shape
    flat = np.ascontiguousarray(reader.embeddings.reshape(n * p, d), dtype=np.float32)
    if os.environ.get("STREAMING_EVAL_BACKEND", "faiss") == "torch":
        patch_scores = _torch_patch_scores(
            bank, flat, min(n_nearest_neighbours, max(len(bank), 1))
        )
    else:
        scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=min(n_nearest_neighbours, max(len(bank), 1)),
            nn_method=patchcore.common.FaissNN(False, _available_cpus()),
        )
        bank.install_into(scorer)
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
    device: Optional[str] = None,
) -> dict:
    """Compute image AUROC (+ pixel AUROC / PRO when masks are available).

    ``reader`` is an :class:`EmbeddingCacheReader` or synthetic ``ArrayReader``
    with ``embeddings``, ``labels``, and optionally ``masks``.
    ``device=None`` auto-selects (GPU when available); it only affects the
    segmentation rescaler, not the k-NN scoring backend.
    """
    from patchcore.streaming.bank import knn_device

    device = str(device) if device else str(knn_device())
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
