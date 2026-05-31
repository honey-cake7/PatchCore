import logging

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from sklearn.svm import LinearSVC

LOGGER = logging.getLogger(__name__)


def select_pseudo_labels(
    score_map,
    percentile=99.0,
    neighbor_radius=1,
    n_nominal=500,
    rng=None,
):
    """Select sparse pseudo-labelled points from an anomaly score map.

    Anomalous points are the local maxima of ``score_map`` whose value exceeds
    the ``percentile``-th percentile ("easy" labels), enriched with their
    spatial neighbours ("hard" labels). Nominal points are uniformly sampled
    from the remaining locations.

    Args:
        score_map: [H, W] float array of anomaly scores.
        percentile: peaks below this percentile of all scores are suppressed.
        neighbor_radius: radius (in pixels) used to enrich surviving peaks.
        n_nominal: number of nominal points to sample.
        rng: optional ``np.random.Generator`` for reproducible nominal sampling.

    Returns:
        (coords, labels): ``coords`` is an [N, 2] int array of (row, col)
        locations, ``labels`` is an [N] int array (1 = anomalous, 0 = nominal).
    """
    if rng is None:
        rng = np.random.default_rng()

    score_map = np.asarray(score_map, dtype=np.float32)
    h, w = score_map.shape

    # Local maxima via neighbour comparison (3x3 window).
    local_max = ndimage.maximum_filter(score_map, size=3, mode="nearest")
    peaks = score_map == local_max

    # Suppress peaks below the percentile threshold -> "easy" anomalous.
    threshold = np.percentile(score_map, percentile)
    peaks &= score_map >= threshold

    # Enrich surviving peaks with spatial neighbours -> "hard" anomalous.
    if neighbor_radius > 0 and peaks.any():
        size = 2 * neighbor_radius + 1
        anomalous_mask = ndimage.maximum_filter(
            peaks.astype(np.uint8), size=size, mode="constant", cval=0
        ).astype(bool)
    else:
        anomalous_mask = peaks

    # Uniformly sample nominal points from the rest of the image.
    nominal_candidates = np.flatnonzero(~anomalous_mask.ravel())
    n_nominal = min(n_nominal, nominal_candidates.size)
    nominal_flat = rng.choice(nominal_candidates, size=n_nominal, replace=False)
    nominal_coords = np.stack(np.unravel_index(nominal_flat, (h, w)), axis=1)

    anomalous_coords = np.argwhere(anomalous_mask)

    coords = np.concatenate([anomalous_coords, nominal_coords], axis=0)
    labels = np.concatenate(
        [
            np.ones(len(anomalous_coords), dtype=np.int64),
            np.zeros(len(nominal_coords), dtype=np.int64),
        ]
    )
    return coords, labels


class TTT4AS:
    """Per-image test-time SVM classifier over a frozen feature map.

    Args:
        patchcore_model: a fitted ``patchcore.patchcore.PatchCore`` instance,
            used both for the WRN-50 feature path and to know the target size.
        feature_extractor: "wrn50" (reuse PatchCore's own features) or
            "dinov2" (load a frozen DINOv2 backbone via torch.hub).
        device: torch device.
        percentile / neighbor_radius / n_nominal: passed to
            :func:`select_pseudo_labels`.
        svm_C: SVM margin regularization (paper uses 0.001).
        seed: base seed for reproducible nominal sampling.
    """

    def __init__(
        self,
        patchcore_model,
        feature_extractor="wrn50",
        device=None,
        percentile=99.0,
        neighbor_radius=1,
        n_nominal=500,
        svm_C=0.001,
        seed=0,
    ):
        self.patchcore = patchcore_model
        self.feature_extractor = feature_extractor
        self.device = device if device is not None else patchcore_model.device
        self.target_size = tuple(patchcore_model.input_shape[-2:])
        self.percentile = percentile
        self.neighbor_radius = neighbor_radius
        self.n_nominal = n_nominal
        self.svm_C = svm_C
        self.seed = seed
        self._dino = None

    
    def extract_feature_map(self, image):
        """Return a dense [H, W, D] feature map for a single image.

        Args:
            image: [1, 3, H, W] (or [3, H, W]) float tensor, already normalized.
        """
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(torch.float).to(self.device)

        if self.feature_extractor == "wrn50":
            return self._extract_wrn50(image)
        if self.feature_extractor == "dinov2":
            return self._extract_dinov2(image)
        raise ValueError(f"Unknown feature_extractor: {self.feature_extractor}")

    def _extract_wrn50(self, image):
        """Reuse PatchCore's own per-patch features (same as the memory bank)."""
        with torch.no_grad():
            features, patch_shapes = self.patchcore._embed(
                image, provide_patch_shapes=True
            )
        features = np.asarray(features)  # [Hf*Wf, D]
        hf, wf = patch_shapes[0]
        d = features.shape[-1]
        grid = torch.from_numpy(features).reshape(1, hf, wf, d).permute(0, 3, 1, 2)
        return self._upsample(grid)

    def _extract_dinov2(self, image):
        """Extract DINOv2 patch tokens and reshape to a spatial grid."""
        if self._dino is None:
            LOGGER.info("Loading DINOv2 (dinov2_vits14) from torch.hub...")
            self._dino = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14"
            )
            self._dino.eval().to(self.device)

        patch = 14
        h, w = image.shape[-2:]
        # DINOv2 requires spatial dims divisible by the patch size.
        h2, w2 = (h // patch) * patch, (w // patch) * patch
        if (h2, w2) != (h, w):
            image = F.interpolate(
                image, size=(h2, w2), mode="bilinear", align_corners=False
            )
        with torch.no_grad():
            out = self._dino.forward_features(image)
            tokens = out["x_norm_patchtokens"]  # [1, Hf*Wf, D]
        hf, wf = h2 // patch, w2 // patch
        d = tokens.shape[-1]
        grid = tokens.reshape(1, hf, wf, d).permute(0, 3, 1, 2)
        return self._upsample(grid)

    def _upsample(self, grid):
        """Bilinearly upsample a [1, D, Hf, Wf] grid to [H, W, D] (numpy)."""
        grid = grid.to(torch.float).to(self.device)
        grid = F.interpolate(
            grid, size=self.target_size, mode="bilinear", align_corners=False
        )
        return grid.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, D]

    def predict_binary_map(self, score_map, feature_map, image_index=0):
        """Train a per-image SVM and predict a dense binary mask.

        Args:
            score_map: [H, W] anomaly score map.
            feature_map: [H, W, D] dense feature map (from extract_feature_map).
            image_index: used to derive a per-image RNG seed.

        Returns:
            [H, W] uint8 binary mask (1 = anomalous, 0 = nominal).
        """
        score_map = np.asarray(score_map, dtype=np.float32)
        h, w = score_map.shape
        rng = np.random.default_rng(self.seed + image_index)

        coords, labels = select_pseudo_labels(
            score_map,
            percentile=self.percentile,
            neighbor_radius=self.neighbor_radius,
            n_nominal=self.n_nominal,
            rng=rng,
        )

        # Degenerate case: no anomalous (or no nominal) pseudo-labels selected.
        # Fall back to a plain percentile threshold on the score map.
        if len(np.unique(labels)) < 2:
            threshold = np.percentile(score_map, self.percentile)
            return (score_map >= threshold).astype(np.uint8)

        train_feats = feature_map[coords[:, 0], coords[:, 1]]  # [N, D]
        clf = LinearSVC(C=self.svm_C, dual="auto")
        clf.fit(train_feats, labels)

        dense = clf.predict(feature_map.reshape(h * w, -1))
        return dense.reshape(h, w).astype(np.uint8)
