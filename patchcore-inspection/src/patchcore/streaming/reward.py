"""Label-free proxy reward for memory-bank maintenance.

The reward never sees anomaly labels. It rewards a bank that (a) keeps recent
normal patches close (low false-positive risk), (b) does not waste budget on
near-duplicates, and (c) keeps the anomaly-score scale stable over time (so a
deployed threshold stays calibrated):

    r_t = -(alpha * C_t + beta * R_t + gamma * U_t)

* C_t coverage : mean NN-1 distance from a held-out recent-normal window to the
  bank. The held-out patches are *never* admissible, which forbids the
  degenerate "admit exactly the probes" solution.
* R_t redundancy : how clumped the bank is below a scale delta.
* U_t instability : maintenance churn plus the mean absolute change in a fixed
  probe set's NN-1 distances between consecutive steps (score-scale drift).
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


def _default_churn_coef() -> float:
    # Own knob (override with STREAMING_CHURN_COEF) so churn can be penalized
    # independently of score-drift. The prior design folded churn into the
    # instability term at gamma=0.1, which was ~0.006 of reward per step at
    # M=20000 — far too weak to deter a policy that rewrites the whole bank
    # every step. Only churn ABOVE churn_budget is penalized (see below), so
    # ordinary maintenance is free; the coefficient prices wholesale rewrites.
    return float(os.environ.get("STREAMING_CHURN_COEF", "2.0"))


def _default_churn_budget() -> float:
    # Per-step churn (admit+evict as a fraction of capacity) below this budget
    # is free. A flat linear churn tax made low-churn-but-inaccurate policies
    # (random/reservoir) out-score every adaptive policy on the proxy while
    # losing on AUROC — the exact misalignment RL then optimized into. 0.01
    # covers fifo/greedy-coreset-scale maintenance; a full periodic re-coreset
    # (churn ≈ 2.0 on its rewrite step) still pays. Override with
    # STREAMING_CHURN_BUDGET.
    return float(os.environ.get("STREAMING_CHURN_BUDGET", "0.01"))


@dataclass
class RewardConfig:
    alpha: float = 1.0
    beta: float = 0.3
    gamma: float = 0.1                         # weight on score-drift (stability)
    churn_coef: float = field(default_factory=_default_churn_coef)
    churn_budget: float = field(default_factory=_default_churn_budget)
    window_images: int = 32
    holdout_patch_frac: float = 0.1
    redundancy_delta: Optional[float] = None  # auto-set from warmup if None
    redundancy_sample: int = 2048
    coverage_scale: float = 1.0               # normalizer for C (set from warmup)


def load_reward_weights(path: str) -> "RewardConfig":
    """Build a RewardConfig from a fitted-weights JSON (bin/fit_reward_weights.py).

    Accepts either the fit output ({"recommended": {...}}) or a bare weights
    dict. Unknown keys are ignored; missing keys keep their defaults. Callers
    must hand each env its own copy (``dataclasses.replace(cfg)``) because
    ``env.reset()`` writes the warmup scales into the config in place.
    """
    import json

    with open(path) as f:
        obj = json.load(f)
    weights = obj.get("recommended", obj)
    cfg = RewardConfig()
    for key in ("alpha", "beta", "gamma", "churn_coef", "churn_budget"):
        if key in weights:
            setattr(cfg, key, float(weights[key]))
    return cfg


def coverage(bank, patches: np.ndarray) -> float:
    """Mean NN-1 distance from ``patches`` [n, D] to the bank (lower is better)."""
    if len(patches) == 0 or len(bank) == 0:
        return 0.0
    dists, _ = bank.knn(np.ascontiguousarray(patches, dtype=np.float32), k=1)
    d = dists[:, 0]
    d = d[np.isfinite(d)]
    return float(d.mean()) if len(d) else 0.0


def redundancy(bank, delta: float, sample: int = 2048) -> float:
    """Fraction-weighted clumping of bank members below scale ``delta``."""
    nn2 = bank.member_redundancy(sample=sample)
    if len(nn2) == 0 or delta <= 0:
        return 0.0
    return float(np.clip(1.0 - nn2 / delta, 0.0, None).mean())


class ProxyReward:
    """Stateful reward: tracks a fixed probe set for the stability term."""

    def __init__(self, cfg: RewardConfig, probe: np.ndarray) -> None:
        self.cfg = cfg
        self.probe = np.ascontiguousarray(probe, dtype=np.float32)
        self._prev_probe_dists: Optional[np.ndarray] = None

    def _probe_dists(self, bank) -> np.ndarray:
        if len(self.probe) == 0 or len(bank) == 0:
            return np.zeros(len(self.probe), dtype=np.float32)
        d, _ = bank.knn(self.probe, k=1)
        return np.where(np.isfinite(d[:, 0]), d[:, 0], 0.0)

    def components(
        self, bank, holdout_patches: np.ndarray, n_admit: int, n_evict: int
    ) -> Tuple[float, float, float, float]:
        cfg = self.cfg
        c = coverage(bank, holdout_patches) / (cfg.coverage_scale + 1e-8)
        delta = cfg.redundancy_delta if cfg.redundancy_delta else cfg.coverage_scale
        r = redundancy(bank, delta, cfg.redundancy_sample)
        churn = (n_admit + n_evict) / max(bank.capacity, 1)
        probe_dists = self._probe_dists(bank)
        if self._prev_probe_dists is not None and len(probe_dists) == len(
            self._prev_probe_dists
        ):
            score_drift = float(
                np.abs(probe_dists - self._prev_probe_dists).mean()
            ) / (cfg.coverage_scale + 1e-8)
        else:
            score_drift = 0.0
        self._prev_probe_dists = probe_dists
        return c, r, churn, score_drift

    def compute(
        self, bank, holdout_patches: np.ndarray, n_admit: int, n_evict: int
    ) -> Tuple[float, Dict[str, float]]:
        c, r, churn, score_drift = self.components(
            bank, holdout_patches, n_admit, n_evict
        )
        cfg = self.cfg
        churn_excess = max(0.0, churn - cfg.churn_budget)
        reward = -(
            cfg.alpha * c
            + cfg.beta * r
            + cfg.gamma * score_drift
            + cfg.churn_coef * churn_excess
        )
        # U kept for logging continuity (instability = churn + score-drift).
        # Raw churn stays in the log so offline reward-weight refits can
        # re-derive churn_excess under any candidate budget.
        return reward, {
            "C": c, "R": r, "U": churn + score_drift,
            "churn": churn, "churn_excess": churn_excess,
            "score_drift": score_drift, "reward": reward,
        }


def estimate_scales(bank, sample_patches: np.ndarray) -> Tuple[float, float]:
    """Warmup helper: (coverage_scale, redundancy_delta) from a reference bank.

    ``redundancy_delta`` is the median bank-member NN-2 distance (the manifold's
    natural inter-sample spacing). ``coverage_scale`` is the median NN-1 distance
    of *held-out* sample patches to the bank — but when the sample overlaps the
    bank (many exact members → distance 0), the median collapses; we then fall
    back to the member spacing ``delta``. Both make the reward terms O(1) and
    dataset-agnostic.
    """
    nn2 = bank.member_redundancy(sample=min(2048, len(bank)))
    delta = float(np.median(nn2)) if len(nn2) else 1.0
    delta = max(delta, 1e-6)

    dists, _ = bank.knn(np.ascontiguousarray(sample_patches, dtype=np.float32), k=1)
    d = dists[:, 0]
    d = d[np.isfinite(d)]
    # Exact bank members must count as distance 0. Brute-force L2 via matmul
    # (torch/faiss alike) returns O(1e-3)-scale noise for identical vectors,
    # so use a threshold relative to the member spacing, not an absolute one.
    positive = d[d > max(1e-6, 0.05 * delta)]
    if len(positive) >= max(4, int(0.2 * len(d))):
        cov_scale = float(np.median(positive))
    else:
        cov_scale = delta  # sample is essentially a subset of the bank
    return max(cov_scale, 1e-6), delta
