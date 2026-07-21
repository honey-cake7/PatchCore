"""The memory-maintenance MDP: one env step = one incoming normal image.

State summarizes bank + stream statistics (never raw images); the action is a
compact admission-bar + eviction-utility decision; the transition is
deterministic given the cached stream; the reward is the label-free proxy on a
held-out slice of the recent window. Both the learned policy and the hand-designed
baselines drive the same transition via :meth:`step_with_decision`.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np

from patchcore.streaming.bank import (
    DynamicMemoryBank,
    intra_batch_nn2,
    knn_device,
    pinned_global_seed,
)
from patchcore.streaming.reward import ProxyReward, RewardConfig, estimate_scales


OBS_DIM = 53


@dataclass
class ObsConfig:
    d_proj: int = 16
    window_images: int = 32
    ewma_lambda: float = 0.05
    slope_horizon: int = 20
    warmup_steps: int = 128        # steps over which RunningNorm adapts, then freezes


@dataclass
class ActionConfig:
    mode: str = "continuous6"      # "continuous6" | "bar_only" | "discrete4"
    admit_cap_frac: float = 0.05   # max admissions per step, as a fraction of M
    discrete_k: int = 8            # patches added/evicted per discrete action


class RunningNorm:
    """Welford mean/var normalizer; adapts during warmup then freezes."""

    def __init__(self, dim: int) -> None:
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)
        self.count = 0
        self.frozen = False

    def update(self, x: np.ndarray) -> None:
        if self.frozen:
            return
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.count < 2:
            return x
        std = np.sqrt(self.m2 / (self.count - 1)) + 1e-6
        return ((x - self.mean) / std).astype(np.float32)

    def freeze(self) -> None:
        self.frozen = True

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.copy(),
            "m2": self.m2.copy(),
            "count": int(self.count),
            "frozen": bool(self.frozen),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "RunningNorm":
        norm = cls(len(state["mean"]))
        norm.mean = np.asarray(state["mean"], dtype=np.float64).copy()
        norm.m2 = np.asarray(state["m2"], dtype=np.float64).copy()
        norm.count = int(state["count"])
        norm.frozen = bool(state["frozen"])
        return norm


class MemoryMaintenanceEnv:
    def __init__(
        self,
        reader,
        capacity: int,
        reward_cfg: Optional[RewardConfig] = None,
        obs_cfg: Optional[ObsConfig] = None,
        action_cfg: Optional[ActionConfig] = None,
        seed: int = 0,
        init_bank: str = "coreset_stage0",
        warmup_images: int = 100,
        n_nearest_neighbours: int = 1,
        obs_norm: Optional["RunningNorm"] = None,
    ) -> None:
        self.reader = reader
        self.capacity = int(capacity)
        self.dim = reader.dim
        self.reward_cfg = reward_cfg or RewardConfig()
        self.obs_cfg = obs_cfg or ObsConfig()
        self.action_cfg = action_cfg or ActionConfig()
        self.rng = np.random.default_rng(seed)
        self.init_bank_mode = init_bank
        self.warmup_images = warmup_images
        self.n_nn = n_nearest_neighbours

        # Frozen random projection for the observation's distributional summary.
        self._proj = self.rng.normal(
            size=(self.dim, self.obs_cfg.d_proj)
        ).astype(np.float32) / np.sqrt(self.dim)
        # An injected obs_norm (e.g. the one a checkpointed policy trained
        # under, or one shared across vectorized training envs) is used as-is;
        # otherwise each env fits its own during warmup.
        self._norm = obs_norm if obs_norm is not None else RunningNorm(OBS_DIM)
        self._last_obs: Optional[np.ndarray] = None

        self.bank: Optional[DynamicMemoryBank] = None
        self._proxy: Optional[ProxyReward] = None
        # The warmup coreset is expensive; compute it once and restore a snapshot
        # on every subsequent reset instead of re-subsampling each episode.
        self._init_snapshot = None
        self._probe: Optional[np.ndarray] = None
        self._window: Deque[np.ndarray] = deque(maxlen=self.obs_cfg.window_images)
        self._novelty_hist: Deque[float] = deque(maxlen=self.obs_cfg.slope_horizon)
        self._ewma = 0.0
        self._prev_coverage = 0.0
        # (added, evicted) from a replace_bank() call, folded into the next
        # step's churn so wholesale rewrites (periodic-coreset) are metered on
        # the same footing as admit/evict policies.
        self._pending_churn = (0, 0)

    # ---- initialization --------------------------------------------------
    def _warmup_features(self) -> np.ndarray:
        ids = list(range(min(self.warmup_images, self.reader.n_images)))
        return self.reader.flat_slice(ids)

    def _init_bank(self) -> DynamicMemoryBank:
        feats = self._warmup_features()
        if self.init_bank_mode == "coreset_stage0" and len(feats):
            import torch

            import patchcore.sampler

            if len(feats) > self.capacity:
                pct = self.capacity / len(feats)
                sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(
                    percentage=pct, device=knn_device()
                )
                with pinned_global_seed(int(self.rng.integers(2**31))):
                    feats = sampler.run(np.ascontiguousarray(feats, dtype=np.float32))
            return DynamicMemoryBank.from_vectors(
                feats[: self.capacity], capacity=self.capacity,
            )
        return DynamicMemoryBank(self.capacity, self.dim)

    def reset(self, episode_slice: Optional[slice] = None) -> np.ndarray:
        if self._init_snapshot is None:
            # First episode: pay the coreset cost once, then cache the result.
            self.bank = self._init_bank()
            self._init_snapshot = self.bank.snapshot()
            warm = self._warmup_features()
            self._probe = warm[
                self.rng.choice(len(warm), size=min(256, len(warm)), replace=False)
            ]
            cov_scale, delta = estimate_scales(self.bank, self._probe)
            self.reward_cfg.coverage_scale = cov_scale
            if self.reward_cfg.redundancy_delta is None:
                self.reward_cfg.redundancy_delta = delta
        else:
            # Later episodes: restore the cached warmup bank (cheap memcpy).
            self.bank = DynamicMemoryBank(self.capacity, self.dim)
            self.bank.restore(self._init_snapshot)
        self._proxy = ProxyReward(self.reward_cfg, self._probe)

        self._window.clear()
        self._novelty_hist.clear()
        self._ewma = 0.0
        self._prev_coverage = 0.0
        self._pending_churn = (0, 0)

        start = self.warmup_images if episode_slice is None else episode_slice.start
        stop = self.reader.n_images if episode_slice is None else episode_slice.stop
        self._start, self._stop = int(start), int(min(stop, self.reader.n_images))
        self._t = self._start
        self._load_current()
        return self._observe()

    # ---- per-step data ---------------------------------------------------
    def _load_current(self) -> None:
        patches = self.reader.image_patches(self._t)
        n_hold = max(1, int(len(patches) * self.reward_cfg.holdout_patch_frac))
        perm = self.rng.permutation(len(patches))
        self._holdout = patches[perm[:n_hold]]
        self._admissible = patches[perm[n_hold:]]
        # batch novelty distances vs current bank (also used by baselines)
        d, _ = self.bank.knn(self._admissible, k=1)
        self._batch_nn = np.where(np.isfinite(d[:, 0]), d[:, 0], 0.0)

    @property
    def current_batch(self) -> np.ndarray:
        return self._admissible

    @property
    def batch_novelty(self) -> np.ndarray:
        return self._batch_nn

    @property
    def stage(self) -> int:
        return self.reader.stage_of(self._t)

    # ---- observation -----------------------------------------------------
    @property
    def last_obs(self) -> Optional[np.ndarray]:
        """The observation most recently returned by reset()/step_with_decision().

        Policies that need the current observation (PPO) must read this instead
        of calling ``_observe()`` again: a second call would advance the
        RunningNorm warmup twice per step, silently changing the normalization
        the policy was trained under (and doubling the obs k-NN cost).
        """
        return self._last_obs

    def _observe(self, update: bool = True) -> np.ndarray:
        cfg = self.obs_cfg
        bank = self.bank
        A = self._admissible
        occ = bank.occupancy
        t_frac = (self._t - self._start) / max(self._stop - self._start, 1)
        since_boundary = len(self._novelty_hist) / max(cfg.slope_horizon, 1)

        nn = self._batch_nn
        nn_log = np.log1p(nn)
        novelty = [nn_log.mean(), nn_log.std(), np.median(nn_log),
                   np.percentile(nn_log, 90), nn_log.max()] if len(nn) else [0]*5

        # intra-batch density: each admissible patch to nearest other patch
        if len(A) > 1:
            intra = intra_batch_nn2(A)
            intra_log = np.log1p(intra)
            density = [intra_log.mean(), np.median(intra_log), np.percentile(intra_log, 90)]
        else:
            density = [0.0, 0.0, 0.0]

        # sliding-window coverage
        if self._window:
            win = np.concatenate(list(self._window), axis=0)
            wd, _ = bank.knn(win, k=1)
            wd = np.where(np.isfinite(wd[:, 0]), wd[:, 0], 0.0)
            cov_mean = wd.mean(); cov_p90 = np.percentile(wd, 90)
        else:
            cov_mean = cov_p90 = 0.0
        cov_delta = cov_mean - self._prev_coverage

        red = bank.member_redundancy(sample=1024)
        red_feat = [red.mean(), np.percentile(red, 10)] if len(red) else [0.0, 0.0]

        slope = 0.0
        if len(self._novelty_hist) >= 2:
            y = np.asarray(self._novelty_hist)
            x = np.arange(len(y))
            slope = float(np.polyfit(x, y, 1)[0])

        # age stats straight from insert steps (entry_features would drag in
        # the O(M^2) redundancy column, which only evictions need)
        if len(bank):
            slots = bank.active_slots()
            age = (bank.step - bank.insert_step[slots]).astype(np.float32)
            age_norm = age / (age.max() + 1e-6)
            age_feat = [age_norm.mean(), age_norm.max()]
        else:
            age_feat = [0.0, 0.0]

        # random-projection distributional summary
        proj_batch = (A @ self._proj).mean(axis=0) if len(A) else np.zeros(cfg.d_proj)
        proj_bank = bank.projected_mean(self._proj)
        proj_diff = np.linalg.norm(proj_batch - proj_bank)

        scalar = np.array(
            [occ, t_frac, since_boundary, *novelty, *density,
             cov_mean, cov_p90, cov_delta, *red_feat, self._ewma, slope, *age_feat],
            dtype=np.float32,
        )
        obs = np.concatenate([scalar, proj_batch, proj_bank, [proj_diff]]).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        if update:
            self._norm.update(obs)
            if self._norm.count >= cfg.warmup_steps:
                self._norm.freeze()
        self._last_obs = self._norm.normalize(obs)
        return self._last_obs

    # ---- action decoding -------------------------------------------------
    def decode_action(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Map a raw policy action to (admit_idx into current_batch, evict_slots)."""
        a = np.tanh(np.asarray(action, dtype=np.float32))
        mode = self.action_cfg.mode
        A = self._admissible
        nn = self._batch_nn
        cap = int(self.action_cfg.admit_cap_frac * self.capacity)

        if mode == "discrete4":
            return self._decode_discrete(int(np.argmax(action)))

        bar = (a[0] + 1.0) / 2.0                          # [0,1]
        if len(nn):
            thresh = np.quantile(nn, bar)
            admit_mask = nn >= thresh
        else:
            admit_mask = np.zeros(0, dtype=bool)
        admit_idx = np.flatnonzero(admit_mask)

        if mode == "continuous6":
            cap_frac = (a[1] + 1.0) / 2.0 * self.action_cfg.admit_cap_frac
            cap = max(1, int(cap_frac * self.capacity))
        # keep the most novel up to the cap
        if len(admit_idx) > cap:
            order = np.argsort(-nn[admit_idx])
            admit_idx = admit_idx[order[:cap]]

        n_admit = len(admit_idx)
        free = self.capacity - len(self.bank)
        n_evict = max(0, n_admit - free)
        evict_slots = self._select_evictions(n_evict, a[2:6] if mode == "continuous6" else None)
        return admit_idx, evict_slots

    def _select_evictions(self, n_evict: int, weights: Optional[np.ndarray]) -> np.ndarray:
        if n_evict <= 0 or len(self.bank) == 0:
            return np.empty(0, dtype=np.int64)
        slots = self.bank.active_slots()
        phi = self.bank.entry_features()  # [m,4], higher = keep
        if weights is None:
            w = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)  # evict most redundant
        else:
            w = np.asarray(weights, dtype=np.float32)
        util = phi @ w
        order = np.argsort(util)  # lowest utility first
        return slots[order[:n_evict]]

    def _decode_discrete(self, choice: int) -> Tuple[np.ndarray, np.ndarray]:
        k = self.action_cfg.discrete_k
        A, nn = self._admissible, self._batch_nn
        top = np.argsort(-nn)[:k] if len(nn) else np.empty(0, dtype=np.int64)
        if choice == 0:                                   # no-op
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        free = self.capacity - len(self.bank)
        n_evict = max(0, len(top) - free)
        if choice == 1:                                   # admit novel, evict oldest
            return top, self._select_evictions(n_evict, np.array([1., 0, 0, 0]))
        if choice == 2:                                   # admit novel, evict redundant
            return top, self._select_evictions(n_evict, np.array([0, 0, 1., 0]))
        # choice 3: admit random, evict random
        ridx = self.rng.choice(len(A), size=min(k, len(A)), replace=False) if len(A) else top
        rev = self.bank.active_slots()
        rev = self.rng.choice(rev, size=min(n_evict, len(rev)), replace=False) if n_evict else np.empty(0, np.int64)
        return ridx, rev

    # ---- transition ------------------------------------------------------
    def step(self, action: np.ndarray):
        admit_idx, evict_slots = self.decode_action(action)
        return self.step_with_decision(admit_idx, evict_slots)

    def step_with_decision(self, admit_idx: np.ndarray, evict_slots: np.ndarray):
        admit_idx = np.asarray(admit_idx, dtype=np.int64)
        evict_slots = np.asarray(evict_slots, dtype=np.int64)
        # enforce budget: never let admissions exceed free capacity after evictions
        self.bank.evict(evict_slots)
        free = self.capacity - len(self.bank)
        if len(admit_idx) > free:
            admit_idx = admit_idx[:free]
        n_admit = len(admit_idx)
        if n_admit:
            self.bank.add(self._admissible[admit_idx], stage=self.stage)

        # fold in any churn from a preceding replace_bank() this step
        pend_admit, pend_evict = self._pending_churn
        self._pending_churn = (0, 0)
        tot_admit = n_admit + pend_admit
        tot_evict = len(evict_slots) + pend_evict

        self._window.append(self._holdout)
        holdout = np.concatenate(list(self._window), axis=0)
        reward, comps = self._proxy.compute(
            self.bank, holdout, tot_admit, tot_evict
        )

        # drift EWMA over batch novelty mean
        nov = float(self._batch_nn.mean()) if len(self._batch_nn) else 0.0
        self._ewma = (1 - self.obs_cfg.ewma_lambda) * self._ewma + self.obs_cfg.ewma_lambda * nov
        self._novelty_hist.append(nov)
        self._prev_coverage = comps["C"] * self.reward_cfg.coverage_scale

        self._t += 1
        done = self._t >= self._stop
        info = {
            "stage": self.stage if not done else self.reader.stage_of(self._t - 1),
            "n_admit": tot_admit, "n_evict": tot_evict,
            "occupancy": self.bank.occupancy, **comps,
        }
        if done:
            return self._observe_terminal(), reward, True, info
        self._load_current()
        return self._observe(), reward, False, info

    def _observe_terminal(self) -> np.ndarray:
        self._t -= 1
        obs = self._observe()
        self._t += 1
        return obs

    def replace_bank(self, vectors: np.ndarray) -> None:
        """Wholesale replace the active bank contents (periodic-coreset baseline).

        Records the (added, evicted) counts so the next step_with_decision folds
        them into that step's churn — otherwise a full rewrite would be free in
        the reward, which unfairly advantages this baseline over admit/evict
        policies.
        """
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)[: self.capacity]
        n_evicted = len(self.bank)
        self.bank.evict(self.bank.active_slots())
        n_added = 0
        if len(vectors):
            self.bank.add(vectors, stage=self.stage)
            n_added = len(vectors)
        prev_add, prev_evict = self._pending_churn
        self._pending_churn = (prev_add + n_added, prev_evict + n_evicted)
