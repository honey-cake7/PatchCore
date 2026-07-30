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
        """Patch features for image ``i`` as an in-memory array ([P, D])."""
        return np.asarray(self.embeddings[i], dtype=np.float32)

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
