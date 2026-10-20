"""On-disk memmap cache of PatchCore patch embeddings.

The frozen backbone runs exactly twice per (dataset, backbone, drift-config):
once over the ordered normal stream, once over the per-stage labeled test sets.
Both are written here as float32 memmaps with a row-major, per-image layout so
that "give me image i's patches" is a single contiguous slice — the streaming
unit consumed by the RL environment.

Layout (one directory per config)::

    <cache_dir>/
        stream/embeddings.npy      float32 [N_img, P, D]
        stream/manifest.json
        test/stage_<s>/embeddings.npy  float32 [N_s, P, D]
        test/stage_<s>/labels.npy      int8    [N_s]
        test/stage_<s>/masks.npy       uint8   [N_s, H, W]
        test/stage_<s>/manifest.json
"""
import json
import os
from typing import Optional, Sequence

import numpy as np


class EmbeddingCacheWriter:
    """Streams per-image patch embeddings into a preallocated memmap."""

    def __init__(
        self,
        out_dir: str,
        n_images: int,
        patches_per_image: int,
        dim: int,
        meta: Optional[dict] = None,
    ) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.n_images = int(n_images)
        self.patches_per_image = int(patches_per_image)
        self.dim = int(dim)
        self.meta = dict(meta or {})
        self._emb = np.lib.format.open_memmap(
            os.path.join(out_dir, "embeddings.npy"),
            mode="w+",
            dtype=np.float32,
            shape=(self.n_images, self.patches_per_image, self.dim),
        )
        self._written = np.zeros(self.n_images, dtype=bool)

    def write(self, image_index: int, feats: np.ndarray) -> None:
        """Store one image's patch features ([P, D])."""
        feats = np.asarray(feats, dtype=np.float32)
        if feats.shape != (self.patches_per_image, self.dim):
            raise ValueError(
                f"expected [{self.patches_per_image}, {self.dim}], got {feats.shape}"
            )
        self._emb[image_index] = feats
        self._written[image_index] = True

    def write_labels(self, labels: Sequence[int]) -> None:
        np.save(
            os.path.join(self.out_dir, "labels.npy"),
            np.asarray(labels, dtype=np.int8),
        )

    def write_masks(self, masks: np.ndarray) -> None:
        np.save(os.path.join(self.out_dir, "masks.npy"), np.asarray(masks, np.uint8))

    def finalize(self) -> None:
        self._emb.flush()
        n_missing = int((~self._written).sum())
        self.meta.update(
            {
                "n_images": self.n_images,
                "patches_per_image": self.patches_per_image,
                "dim": self.dim,
                "n_missing": n_missing,
            }
        )
        with open(os.path.join(self.out_dir, "manifest.json"), "w") as f:
            json.dump(self.meta, f, indent=2, default=str)
        if n_missing:
            raise RuntimeError(f"{n_missing} images were never written to the cache")


class EmbeddingCacheReader:
    """Read-only view over a written embedding cache directory."""

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        self.embeddings = np.load(
            os.path.join(cache_dir, "embeddings.npy"), mmap_mode="r"
        )
        with open(os.path.join(cache_dir, "manifest.json")) as f:
            self.manifest = json.load(f)
        self._stage_of = np.asarray(
            self.manifest.get("stage_ids", [0] * len(self.embeddings)), dtype=np.int64
        )
        labels_path = os.path.join(cache_dir, "labels.npy")
        self.labels = np.load(labels_path) if os.path.exists(labels_path) else None
        masks_path = os.path.join(cache_dir, "masks.npy")
        self.masks = (
            np.load(masks_path, mmap_mode="r")
            if os.path.exists(masks_path)
            else None
        )

    def __len__(self) -> int:
        return len(self.embeddings)

    @property
    def n_images(self) -> int:
        return len(self.embeddings)

    @property
    def patches_per_image(self) -> int:
        return self.embeddings.shape[1]

    @property
    def dim(self) -> int:
        return self.embeddings.shape[2]

    def image_patches(self, i: int) -> np.ndarray:
        """Patch features for image ``i`` as an in-memory array ([P, D]), memoized.

        Vectorized lockstep envs share one reader and sit at the same
        timestep, so each step up to 8 envs independently ask for the same
        image; the memo makes the mmap read happen once instead of once per
        env. Two slots cover the step/reset boundary. The array is
        materialized (not a lazy mmap view); callers must not mutate the
        returned array in place (index it — fancy indexing copies).
        """
        key = int(i)
        memo = getattr(self, "_patches_memo", None)
        if memo is None:
            memo = self._patches_memo = {}
        hit = memo.get(key)
        if hit is not None:
            return hit
        arr = np.ascontiguousarray(self.embeddings[i], dtype=np.float32)
        memo[key] = arr
        while len(memo) > 2:  # dict preserves insertion order: drop oldest
            memo.pop(next(iter(memo)))
        return arr

    def image_patches_dev(self, i: int, device):
        """Patch features for image ``i`` as a device tensor, memoized.

        Vectorized training envs replay the same stream in lockstep, so all of
        them need the same image on the k-NN device at the same step; the memo
        makes the host->device upload happen once instead of once per env. Two
        slots cover the step/reset boundary; callers must not mutate the
        returned tensor in place (index it — gathers copy).
        """
        import torch

        key = (int(i), str(device))
        memo = getattr(self, "_dev_memo", None)
        if memo is None:
            memo = self._dev_memo = {}
        hit = memo.get(key)
        if hit is not None:
            return hit
        t = torch.from_numpy(self.image_patches(i)).to(device)
        memo[key] = t
        while len(memo) > 2:  # dict preserves insertion order: drop oldest
            memo.pop(next(iter(memo)))
        return t

    def stage_of(self, i: int) -> int:
        return int(self._stage_of[i])

    def flat_slice(self, image_ids: Sequence[int]) -> np.ndarray:
        """Concatenated patch features of the given images ([len*P, D])."""
        image_ids = list(image_ids)
        if not image_ids:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.concatenate(
            [self.image_patches(i) for i in image_ids], axis=0
        ).astype(np.float32)


class SubsetReader:
    """A fixed subset of a labeled test reader's images (same interface).

    ``embeddings`` materializes the subset on every access (fancy indexing
    copies), so callers should read it once per evaluation — holding all four
    stage subsets in RAM would cost ~4x a test set.
    """

    def __init__(self, base, idx) -> None:
        self.base = base
        self.idx = np.asarray(sorted(idx), dtype=np.int64)
        self.labels = None if base.labels is None else np.asarray(base.labels)[self.idx]
        self.masks = None if base.masks is None else np.asarray(base.masks[self.idx])
        self.manifest = base.manifest

    @property
    def embeddings(self) -> np.ndarray:
        return self.base.embeddings[self.idx]

    @property
    def n_images(self) -> int:
        return len(self.idx)

    def __len__(self) -> int:
        return len(self.idx)


def calibration_split(labels, frac: float, seed: int = 0):
    """Stratified (calib_idx, eval_idx) split of one labeled test set.

    Every stage's test cache holds the same images in the same order (one
    drifted copy per stage), so one split applied to all stages keeps the
    calibration and evaluation images disjoint at every stage.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    calib = []
    for lab in np.unique(labels):
        ids = rng.permutation(np.flatnonzero(labels == lab))
        calib.extend(ids[: max(1, int(round(frac * len(ids))))].tolist())
    calib = np.asarray(sorted(calib), dtype=np.int64)
    eval_ = np.setdiff1d(np.arange(len(labels)), calib)
    return calib, eval_


def split_test_readers(readers, frac: float, part: str, seed: int = 0):
    """Return the ``part`` ("calib" | "eval") subset of every stage's reader;
    ``frac`` <= 0 returns the readers unchanged (legacy: calibrate on test)."""
    if not frac or frac <= 0:
        return readers
    calib, eval_ = calibration_split(readers[0].labels, frac, seed)
    idx = calib if part == "calib" else eval_
    for r in readers[1:]:
        if len(r.labels) != len(readers[0].labels) or (r.labels != readers[0].labels).any():
            raise ValueError("stage test sets differ — cannot share one split")
    return [SubsetReader(r, idx) for r in readers]


def stage0_end(reader) -> int:
    """Exclusive index of the first post-shift image in a stream cache."""
    later = np.flatnonzero(reader._stage_of > 0)
    return int(later[0]) if len(later) else reader.n_images
