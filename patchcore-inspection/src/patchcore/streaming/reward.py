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
import torch


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
    # Weight on tail coverage (p90 of holdout NN-1 distances): anomaly scores
    # come from the *worst*-covered patches, which the mean can hide.
    c90_coef: float = 0.0
    # Weight on probe retention (mean NN-1 distance of the fixed stage-0 probe
    # set): a label-free forgetting signal. Policies that admit recent patches
    # by evicting the old distribution (streaming-greedy-coreset) look perfect
    # on coverage while collapsing stage-0 AUROC — this term is what separates
    # them from policies that admit recent AND retain (fifo/periodic-coreset).
    probe_coef: float = 0.0
    # Weight on the tail ratio Q = C90/C of holdout scores. AUROC is
    # scale-invariant — it measures normal/anomaly score SEPARATION, which
    # absolute distance levels (C, P) cannot see: a re-coreset that doubles
    # every distance keeps AUROC intact. Q is the label-free scale-invariant
    # stand-in: a heavy right tail of normal scores is what actually erodes
    # AUROC. Empirically Q alone nearly reproduces the policy ranking that no
    # weighting of the absolute terms could.
    q_coef: float = 0.0
    # "level": reward = -(state cost) - (action cost)   [default]
    # "delta": potential-based shaping — reward the step CHANGE in state cost
    #   instead of its level. The episode return telescopes to the same
    #   objective (final-minus-initial state cost plus action costs), so the
    #   optimal policy is preserved, but the per-step signal becomes
    #   action-local: at large M a single decision barely moves the state
    #   terms' level, which buries the advantage signal in state inertia.
    reward_form: str = "level"
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
    for key in ("alpha", "beta", "gamma", "churn_coef", "churn_budget",
                "c90_coef", "probe_coef", "q_coef"):
        if key in weights:
            setattr(cfg, key, float(weights[key]))
    return cfg


def coverage(bank, patches) -> float:
    """Mean NN-1 distance from ``patches`` [n, D] to the bank (lower is better).

    ``patches`` may be numpy or a torch tensor already on the bank's device.
    """
    if len(patches) == 0 or len(bank) == 0:
        return 0.0
    if not isinstance(patches, torch.Tensor):
        patches = np.ascontiguousarray(patches, dtype=np.float32)
    dists, _ = bank.knn(patches, k=1)
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
        # The probe set is fixed for the life of this reward; cache it on the
        # k-NN device so the per-step probe query skips the host->device upload.
        self._probe_dev: Optional[torch.Tensor] = None
        self._prev_probe_dists: Optional[np.ndarray] = None
        self._prev_potential_cost: Optional[float] = None

    def reset_state(self) -> None:
        """Clear per-episode state (envs reuse the reward across resets to
        keep the device-cached probe; the step-to-step deltas must not leak
        across episode boundaries)."""
        self._prev_probe_dists = None
        self._prev_potential_cost = None

    def _probe_dists(self, bank) -> np.ndarray:
        if len(self.probe) == 0 or len(bank) == 0:
            return np.zeros(len(self.probe), dtype=np.float32)
        if self._probe_dev is None or self._probe_dev.device != bank.device:
            self._probe_dev = torch.from_numpy(self.probe).to(bank.device)
        d, _ = bank.knn(self._probe_dev, k=1)
        return np.where(np.isfinite(d[:, 0]), d[:, 0], 0.0)

    def components(
        self, bank, holdout_patches: np.ndarray, n_admit: int, n_evict: int
    ) -> Tuple[float, float, float, float, float, float]:
        """Returns (C, R, churn, score_drift, C90, P)."""
        cfg = self.cfg
        scale = cfg.coverage_scale + 1e-8
        # holdout coverage: mean + p90 tail from one k-NN call
        if len(holdout_patches) and len(bank):
            if not isinstance(holdout_patches, torch.Tensor):
                holdout_patches = np.ascontiguousarray(
                    holdout_patches, dtype=np.float32)
            dists, _ = bank.knn(holdout_patches, k=1)
            d = dists[:, 0]
            d = d[np.isfinite(d)]
            c = float(d.mean()) / scale if len(d) else 0.0
            c90 = float(np.percentile(d, 90)) / scale if len(d) else 0.0
        else:
            c = c90 = 0.0
        delta = cfg.redundancy_delta if cfg.redundancy_delta else cfg.coverage_scale
        r = redundancy(bank, delta, cfg.redundancy_sample)
        churn = (n_admit + n_evict) / max(bank.capacity, 1)
        probe_dists = self._probe_dists(bank)
        # probe retention: the LEVEL of probe distances (label-free forgetting
        # signal); score_drift is their step-to-step CHANGE (scale stability).
        p = float(probe_dists.mean()) / scale if len(probe_dists) else 0.0
        if self._prev_probe_dists is not None and len(probe_dists) == len(
            self._prev_probe_dists
        ):
            score_drift = float(
                np.abs(probe_dists - self._prev_probe_dists).mean()
            ) / scale
        else:
            score_drift = 0.0
        self._prev_probe_dists = probe_dists
        return c, r, churn, score_drift, c90, p

    def compute(
        self, bank, holdout_patches: np.ndarray, n_admit: int, n_evict: int
    ) -> Tuple[float, Dict[str, float]]:
        c, r, churn, score_drift, c90, p = self.components(
            bank, holdout_patches, n_admit, n_evict
        )
        cfg = self.cfg
        churn_excess = max(0.0, churn - cfg.churn_budget)
        q = c90 / c if c > 1e-8 else 0.0
        # state-quality cost (function of the bank state only) vs action costs
        # (score_drift is already a step delta; churn is per-action)
        potential_cost = (
            cfg.alpha * c + cfg.beta * r + cfg.c90_coef * c90
            + cfg.probe_coef * p + cfg.q_coef * q
        )
        action_cost = cfg.gamma * score_drift + cfg.churn_coef * churn_excess
        if cfg.reward_form == "delta":
            prev = self._prev_potential_cost
            state_term = 0.0 if prev is None else potential_cost - prev
            self._prev_potential_cost = potential_cost
            reward = -(state_term + action_cost)
        else:
            reward = -(potential_cost + action_cost)
        # U kept for logging continuity (instability = churn + score-drift).
        # Raw churn stays in the log so offline reward-weight refits can
        # re-derive churn_excess under any candidate budget.
        return reward, {
            "C": c, "R": r, "U": churn + score_drift,
            "churn": churn, "churn_excess": churn_excess,
            "score_drift": score_drift, "C90": c90, "P": p, "Q": q,
            "potential_cost": potential_cost, "reward": reward,
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
