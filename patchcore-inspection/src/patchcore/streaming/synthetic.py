"""Synthetic drifting-Gaussian embedding streams for dataset-free development.

These fixtures duck-type :class:`patchcore.streaming.cache.EmbeddingCacheReader`
(``image_patches``, ``stage_of``, ``n_images``, ``patches_per_image``, ``dim``,
``flat_slice``, plus ``labels``/``masks`` for test readers) so the entire
streaming/RL stack can be exercised on a laptop with no real data.

Normal patch features live on a low-variance mixture-of-Gaussians manifold whose
component means shift from stage to stage (the injected drift). Test "anomalies"
have a fraction of their patches drawn from an off-manifold Gaussian, so their
nearest-neighbour distance to a well-maintained bank is larger — exactly the
signal PatchCore scores.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class SyntheticConfig:
    dim: int = 64
    patches_per_image: int = 16
    n_components: int = 6              # clusters on the normal manifold
    manifold_std: float = 0.35        # within-cluster spread
    n_stages: int = 4
    images_per_stage: int = 200
    drift_strength: float = 2.5       # per-stage mean shift magnitude
    drift_jitter: float = 0.4         # per-image drift variation (heterogeneous acquisition)
    drift_shape: str = "abrupt"       # "abrupt" | "gradual" | "cyclic"
    anomaly_dist: float = 2.8         # off-manifold anomaly displacement
    anomaly_patch_frac: float = 0.25  # fraction of an anomaly image's patches off-manifold
    seed: int = 0
    means: Optional[np.ndarray] = field(default=None, repr=False)
    drift_dirs: Optional[np.ndarray] = field(default=None, repr=False)


class ArrayReader:
    """In-memory reader duck-typing :class:`EmbeddingCacheReader`."""

    def __init__(
        self,
        embeddings: np.ndarray,
        stage_ids: np.ndarray,
        labels: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
    ) -> None:
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self._stage_of = np.asarray(stage_ids, dtype=np.int64)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int8)
        self.masks = None if masks is None else np.asarray(masks, dtype=np.uint8)
        self.manifest = {"synthetic": True}

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
        return np.asarray(self.embeddings[i], dtype=np.float32)

    def stage_of(self, i: int) -> int:
        return int(self._stage_of[i])

    def flat_slice(self, image_ids) -> np.ndarray:
        image_ids = list(image_ids)
        if not image_ids:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.concatenate(
            [self.image_patches(i) for i in image_ids], axis=0
        ).astype(np.float32)


def _stage_offset(cfg: SyntheticConfig, stage: int) -> np.ndarray:
    """Deterministic per-stage displacement of the manifold, per drift shape."""
    if cfg.drift_shape == "abrupt":
        scale = stage
    elif cfg.drift_shape == "gradual":
        scale = stage / max(cfg.n_stages - 1, 1) * (cfg.n_stages - 1)
    elif cfg.drift_shape == "cyclic":
        scale = np.sin(2 * np.pi * stage / max(cfg.n_stages, 1)) * (cfg.n_stages - 1)
    else:
        raise ValueError(f"unknown drift_shape {cfg.drift_shape!r}")
    return cfg.drift_strength * scale * cfg.drift_dirs


def _init_geometry(cfg: SyntheticConfig) -> SyntheticConfig:
    if cfg.means is None or cfg.drift_dirs is None:
        rng = np.random.default_rng(cfg.seed)
        cfg.means = rng.normal(size=(cfg.n_components, cfg.dim)).astype(np.float32) * 3.0
        # A single shared drift direction per component keeps drift interpretable.
        dirs = rng.normal(size=(cfg.dim,)).astype(np.float32)
        cfg.drift_dirs = dirs / (np.linalg.norm(dirs) + 1e-8)
    return cfg


def _sample_image(cfg, rng, stage, anomalous=False):
    """Draw one image's [P, D] patch features."""
    offset = _stage_offset(cfg, stage)
    # Per-image drift variation: real acquisition drift is heterogeneous across
    # images, which spreads normal anomaly-scores and lets subtle anomalies
    # overlap a stale bank's normals — this is the headroom a policy recovers.
    if cfg.drift_jitter and stage > 0:
        img_dir = rng.normal(size=cfg.dim).astype(np.float32)
        img_dir /= np.linalg.norm(img_dir) + 1e-8
        offset = offset + img_dir * cfg.drift_jitter * cfg.drift_strength * rng.uniform(
            0.0, float(stage)
        )
    comp = rng.integers(cfg.n_components)
    # A per-image dominant component gives intra-image density structure.
    weights = np.full(cfg.n_components, 0.15 / cfg.n_components)
    weights[comp] = 1.0 - 0.15
    weights /= weights.sum()
    assign = rng.choice(cfg.n_components, size=cfg.patches_per_image, p=weights)
    centers = cfg.means[assign] + offset
    patches = centers + rng.normal(
        size=(cfg.patches_per_image, cfg.dim)
    ).astype(np.float32) * cfg.manifold_std
    if anomalous:
        n_anom = max(1, int(cfg.patches_per_image * cfg.anomaly_patch_frac))
        idx = rng.choice(cfg.patches_per_image, size=n_anom, replace=False)
        direction = rng.normal(size=cfg.dim).astype(np.float32)
        direction /= np.linalg.norm(direction) + 1e-8
        patches[idx] += direction * cfg.anomaly_dist
    return patches.astype(np.float32)


def make_synthetic_stream(cfg: SyntheticConfig) -> ArrayReader:
    """Ordered normal stream across ``cfg.n_stages`` drift stages."""
    cfg = _init_geometry(cfg)
    rng = np.random.default_rng(cfg.seed + 1)
    embeddings, stages = [], []
    for stage in range(cfg.n_stages):
        for _ in range(cfg.images_per_stage):
            embeddings.append(_sample_image(cfg, rng, stage, anomalous=False))
            stages.append(stage)
    return ArrayReader(np.stack(embeddings), np.asarray(stages))


def make_synthetic_test(
    cfg: SyntheticConfig, stage: int, n_normal: int = 100, n_anomaly: int = 100
) -> ArrayReader:
    """Labeled test reader for one stage: normal + off-manifold anomaly images."""
    cfg = _init_geometry(cfg)
    rng = np.random.default_rng(cfg.seed + 1000 + stage)
    embeddings, labels = [], []
    for _ in range(n_normal):
        embeddings.append(_sample_image(cfg, rng, stage, anomalous=False))
        labels.append(0)
    for _ in range(n_anomaly):
        embeddings.append(_sample_image(cfg, rng, stage, anomalous=True))
        labels.append(1)
    stages = np.full(len(embeddings), stage, dtype=np.int64)
    return ArrayReader(np.stack(embeddings), stages, labels=np.asarray(labels))


def make_all_test_stages(cfg: SyntheticConfig, **kwargs) -> List[ArrayReader]:
    return [make_synthetic_test(cfg, s, **kwargs) for s in range(cfg.n_stages)]
