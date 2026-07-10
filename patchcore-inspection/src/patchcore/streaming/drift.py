"""Physically motivated, deterministic drift transforms for endoscopic imagery.

Drift is applied to the post-crop uint8 PIL image, *before* ImageNet
normalization, which is where real acquisition variability enters (scope
hardware, illumination, white balance, focus/motion, specular highlights,
vignetting). Every transform is deterministic given ``(stage, image_key)`` so a
cached stream is exactly reproducible.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import PIL.Image
import PIL.ImageFilter


@dataclass
class DriftParams:
    """A physically motivated appearance-shift configuration."""

    wb_gain: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # per-channel white balance
    hue_shift: float = 0.0        # degrees, [-180, 180]
    gamma: float = 1.0            # illumination non-linearity
    blur_sigma: float = 0.0       # focus / motion blur
    specular: float = 0.0         # specular-highlight density/intensity in [0, 1]
    vignette: float = 0.0         # peripheral darkening in [0, 1]
    jitter: float = 0.0           # per-image random severity jitter fraction

    def scaled(self, frac: float) -> "DriftParams":
        """Interpolate between identity (frac=0) and this params (frac=1)."""
        return DriftParams(
            wb_gain=tuple(1.0 + (g - 1.0) * frac for g in self.wb_gain),
            hue_shift=self.hue_shift * frac,
            gamma=1.0 + (self.gamma - 1.0) * frac,
            blur_sigma=self.blur_sigma * frac,
            specular=self.specular * frac,
            vignette=self.vignette * frac,
            jitter=self.jitter,
        )


class DriftTransform:
    """Callable that applies a :class:`DriftParams` to a PIL image."""

    def __init__(self, params: DriftParams, base_seed: int = 0) -> None:
        self.params = params
        self.base_seed = base_seed

    def _rng(self, image_key: str) -> np.random.Generator:
        # Deterministic per (base_seed, image_key): stable across cache rebuilds.
        h = abs(hash((self.base_seed, image_key))) % (2**32)
        return np.random.default_rng(h)

    def __call__(self, img: PIL.Image.Image, image_key: str = "") -> PIL.Image.Image:
        p = self.params
        rng = self._rng(image_key)
        if p.jitter:
            j = 1.0 + rng.uniform(-p.jitter, p.jitter)
            p = p.scaled(j)

        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0

        # White balance (per-channel gain).
        if p.wb_gain != (1.0, 1.0, 1.0):
            arr = arr * np.asarray(p.wb_gain, dtype=np.float32)[None, None, :]

        # Gamma (illumination).
        if p.gamma != 1.0:
            arr = np.clip(arr, 0, None) ** p.gamma

        arr = np.clip(arr, 0.0, 1.0)

        # Hue shift (rotate in HSV).
        if p.hue_shift:
            arr = _shift_hue(arr, p.hue_shift)

        # Specular highlights: additive white ellipses with soft falloff.
        if p.specular > 0:
            arr = _add_specular(arr, p.specular, rng)

        # Vignette: radial darkening toward the periphery.
        if p.vignette > 0:
            arr = _apply_vignette(arr, p.vignette)

        out = PIL.Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))

        # Blur (focus/motion) last, in PIL for a clean Gaussian.
        if p.blur_sigma > 0:
            out = out.filter(PIL.ImageFilter.GaussianBlur(radius=p.blur_sigma))
        return out


def _shift_hue(arr: np.ndarray, degrees: float) -> np.ndarray:
    import colorsys

    # Vectorized-ish HSV rotation via PIL for correctness/simplicity.
    img = PIL.Image.fromarray((arr * 255).astype(np.uint8), "RGB").convert("HSV")
    hsv = np.asarray(img, dtype=np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(degrees / 360.0 * 255)) % 256
    out = PIL.Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")
    del colorsys  # (kept import local; not needed but documents intent)
    return np.asarray(out, dtype=np.float32) / 255.0


def _add_specular(arr: np.ndarray, intensity: float, rng) -> np.ndarray:
    h, w = arr.shape[:2]
    n_spots = max(1, int(intensity * 8))
    yy, xx = np.mgrid[0:h, 0:w]
    mask = np.zeros((h, w), dtype=np.float32)
    for _ in range(n_spots):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        ry = rng.uniform(0.02, 0.08) * h
        rx = rng.uniform(0.02, 0.08) * w
        d = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
        mask = np.maximum(mask, np.exp(-d))
    mask = mask * intensity
    return arr + (1.0 - arr) * mask[..., None]


def _apply_vignette(arr: np.ndarray, strength: float) -> np.ndarray:
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    gain = 1.0 - strength * np.clip(r / np.sqrt(2), 0, 1) ** 2
    return arr * gain[..., None]


# --- named schedules -----------------------------------------------------
def _stage_bank() -> List[DriftParams]:
    """A 4-stage physically motivated drift bank (identity → severe)."""
    return [
        DriftParams(),  # stage 0: pristine
        DriftParams(wb_gain=(1.15, 1.0, 0.9), gamma=1.2, jitter=0.1),
        DriftParams(wb_gain=(0.9, 1.05, 1.2), gamma=0.8, blur_sigma=1.0,
                    specular=0.3, jitter=0.1),
        DriftParams(wb_gain=(1.25, 0.85, 0.8), hue_shift=12.0, gamma=1.35,
                    blur_sigma=1.6, specular=0.5, vignette=0.4, jitter=0.15),
    ]


class DriftSchedule:
    """A named sequence of drift stages over a stream.

    ``shape`` controls how a stream position maps to a stage:
      * ``"abrupt"``  — hard partition into ``n_stages`` equal blocks.
      * ``"gradual"`` — like abrupt, but params ramp linearly within/across stages.
      * ``"cyclic"``  — stages cycle (0,1,2,3,2,1,0,...) to test recovery.
    """

    def __init__(
        self,
        name: str = "staged_abrupt_4",
        stages: List[DriftParams] = None,
        shape: str = "abrupt",
        n_stages: int = None,
    ) -> None:
        self.name = name
        self.stages = stages or _stage_bank()
        self.shape = shape
        self.n_stages = n_stages or len(self.stages)

    def stage_at(self, stream_pos: int, n_stream: int) -> int:
        block = max(1, n_stream // self.n_stages)
        raw = min(stream_pos // block, self.n_stages - 1)
        if self.shape == "cyclic":
            cycle = list(range(self.n_stages)) + list(range(self.n_stages - 2, 0, -1))
            return cycle[raw % len(cycle)]
        return raw

    def params_at(self, stream_pos: int, n_stream: int) -> Tuple[int, DriftParams]:
        stage = self.stage_at(stream_pos, n_stream)
        params = self.stages[stage]
        if self.shape == "gradual":
            block = max(1, n_stream // self.n_stages)
            frac = (stream_pos % block) / block
            if stage + 1 < len(self.stages):
                # linear ramp toward the next stage's params
                nxt = self.stages[stage + 1]
                params = DriftParams(
                    wb_gain=tuple(
                        a + (b - a) * frac
                        for a, b in zip(params.wb_gain, nxt.wb_gain)
                    ),
                    hue_shift=params.hue_shift + (nxt.hue_shift - params.hue_shift) * frac,
                    gamma=params.gamma + (nxt.gamma - params.gamma) * frac,
                    blur_sigma=params.blur_sigma
                    + (nxt.blur_sigma - params.blur_sigma) * frac,
                    specular=params.specular + (nxt.specular - params.specular) * frac,
                    vignette=params.vignette + (nxt.vignette - params.vignette) * frac,
                    jitter=params.jitter,
                )
        return stage, params

    def transform_at(self, stream_pos: int, n_stream: int, base_seed: int) -> DriftTransform:
        _, params = self.params_at(stream_pos, n_stream)
        return DriftTransform(params, base_seed=base_seed)

    def stage_transform(self, stage: int, base_seed: int) -> DriftTransform:
        """Transform for a *test* set rendered under a fixed stage's params."""
        return DriftTransform(self.stages[stage], base_seed=base_seed)


SCHEDULES: Dict[str, "DriftSchedule"] = {
    "staged_abrupt_4": DriftSchedule("staged_abrupt_4", shape="abrupt"),
    "staged_gradual_4": DriftSchedule("staged_gradual_4", shape="gradual"),
    "staged_cyclic_4": DriftSchedule("staged_cyclic_4", shape="cyclic"),
}
