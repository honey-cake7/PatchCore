"""Maintenance policies: hand-designed baselines and the learned PPO wrapper.

Every policy drives the same environment transition through
:meth:`MemoryMaintenanceEnv.step_with_decision`, so all policies share budget,
reward logging, and evaluation hooks — only the admit/evict decision differs.
The common interface is ``step(env) -> (reward, done, info)``.
"""
import abc
from typing import Tuple

import numpy as np


def _evict_by_key(env, n_evict: int, key: np.ndarray) -> np.ndarray:
    """Return the ``n_evict`` active slots with the smallest ``key`` value."""
    if n_evict <= 0 or len(env.bank) == 0:
        return np.empty(0, dtype=np.int64)
    slots = env.bank.active_slots()
    order = np.argsort(key)
    return slots[order[:n_evict]]


class BasePolicy(abc.ABC):
    name = "base"

    @abc.abstractmethod
    def decide(self, env) -> Tuple[np.ndarray, np.ndarray]:
        """Return (admit_idx into env.current_batch, evict_slots)."""

    def step(self, env):
        admit_idx, evict_slots = self.decide(env)
        return env.step_with_decision(admit_idx, evict_slots)


class StaticPolicy(BasePolicy):
    name = "static"

    def decide(self, env):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)


class FIFOPolicy(BasePolicy):
    name = "fifo"

    def __init__(self, k: int = 8):
        self.k = k

    def decide(self, env):
        nn = env.batch_novelty
        admit = np.argsort(-nn)[: self.k] if len(nn) else np.empty(0, dtype=np.int64)
        free = env.capacity - len(env.bank)
        n_evict = max(0, len(admit) - free)
        slots = env.bank.active_slots()
        key = env.bank.insert_step[slots] if len(slots) else np.empty(0)
        return admit, _evict_by_key(env, n_evict, key)


class RandomPolicy(BasePolicy):
    name = "random"

    def __init__(self, k: int = 8, seed: int = 0):
        self.k = k
        self.rng = np.random.default_rng(seed)

    def decide(self, env):
        A = env.current_batch
        admit = (
            self.rng.choice(len(A), size=min(self.k, len(A)), replace=False)
            if len(A) else np.empty(0, dtype=np.int64)
        )
        free = env.capacity - len(env.bank)
        n_evict = max(0, len(admit) - free)
        slots = env.bank.active_slots()
        rev = (
            self.rng.choice(slots, size=min(n_evict, len(slots)), replace=False)
            if n_evict else np.empty(0, dtype=np.int64)
        )
        return admit, rev


class ReservoirPolicy(BasePolicy):
    """Vitter reservoir sampling over the incoming patch stream."""

    name = "reservoir"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.n_seen = 0

    def decide(self, env):
        A = env.current_batch
        M = env.capacity
        admit, evict = [], []
        slots = list(env.bank.active_slots())
        free = M - len(env.bank)
        for i in range(len(A)):
            self.n_seen += 1
            if free > 0:
                admit.append(i)
                free -= 1
            else:
                # replace a random reservoir slot with prob M / n_seen
                if self.rng.random() < M / self.n_seen and slots:
                    victim = int(self.rng.choice(len(slots)))
                    evict.append(slots.pop(victim))
                    admit.append(i)
        return np.asarray(admit, dtype=np.int64), np.asarray(evict, dtype=np.int64)


class StreamingGreedyCoresetPolicy(BasePolicy):
    """Admit a patch if it is novel enough; evict the most-redundant member.

    The strongest hand-designed competitor: ``tau`` is a coverage radius adapted
    by an EWMA of batch novelty, so the admission bar tracks drift without any
    per-deployment retuning of a fixed threshold.
    """

    name = "streaming_greedy_coreset"

    def __init__(self, tau_scale: float = 1.0, ewma_lambda: float = 0.05, cap_frac: float = 0.05):
        self.tau_scale = tau_scale
        self.ewma_lambda = ewma_lambda
        self.cap_frac = cap_frac
        self._tau = None

    def decide(self, env):
        nn = env.batch_novelty
        if len(nn) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        med = float(np.median(nn))
        self._tau = med if self._tau is None else (
            (1 - self.ewma_lambda) * self._tau + self.ewma_lambda * med
        )
        admit = np.flatnonzero(nn > self._tau * self.tau_scale)
        cap = max(1, int(self.cap_frac * env.capacity))
        if len(admit) > cap:
            admit = admit[np.argsort(-nn[admit])[:cap]]
        free = env.capacity - len(env.bank)
        n_evict = max(0, len(admit) - free)
        # evict most redundant: entry_features col 2 is -redundancy (higher=keep)
        phi = env.bank.entry_features()
        key = phi[:, 2] if len(phi) else np.empty(0)
        return admit, _evict_by_key(env, n_evict, key)


class PeriodicCoresetPolicy(BasePolicy):
    """Buffer incoming patches; every ``period`` steps re-coreset bank ∪ buffer."""

    name = "periodic_coreset"

    def __init__(self, period: int = 50, device: str = "cpu"):
        self.period = period
        self.device = device
        self._buffer = []
        self._count = 0

    def step(self, env):
        import torch

        import patchcore.sampler

        self._buffer.append(env.current_batch.copy())
        self._count += 1
        if self._count % self.period == 0:
            pool = np.concatenate([env.bank.vectors()] + self._buffer, axis=0)
            if len(pool) > env.capacity:
                pct = env.capacity / len(pool)
                pool = patchcore.sampler.ApproximateGreedyCoresetSampler(
                    percentage=pct, device=torch.device(self.device)
                ).run(np.ascontiguousarray(pool, dtype=np.float32))
            env.replace_bank(pool)
            self._buffer = []
        return env.step_with_decision(np.empty(0, np.int64), np.empty(0, np.int64))

    def decide(self, env):  # unused (step is overridden)
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)


class PPOPolicy(BasePolicy):
    """Wraps a trained ActorCritic; acts with the deterministic mean action."""

    name = "ppo"

    def __init__(self, actor_critic, device: str = "cpu"):
        self.ac = actor_critic
        self.device = device

    def decide(self, env):
        import torch

        obs = env._observe()
        with torch.no_grad():
            t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            dist, _ = self.ac(t)
            action = dist.mean.squeeze(0).cpu().numpy()
        return env.decode_action(action)


BASELINES = {
    p.name: p
    for p in [StaticPolicy, FIFOPolicy, RandomPolicy, ReservoirPolicy,
              StreamingGreedyCoresetPolicy, PeriodicCoresetPolicy]
}


def run_policy(env, policy, per_stage_eval=None):
    """Drive ``policy`` over the full stream; optionally evaluate at stage boundaries.

    ``per_stage_eval`` is a callable ``(env, stage) -> metrics`` invoked whenever
    the stream stage changes (and at the end). Returns a trajectory summary.
    """
    env.reset()
    rewards, infos, evals = [], [], []
    prev_stage = env.stage
    done = False
    while not done:
        stage_before = env.stage
        _, reward, done, info = policy.step(env)
        rewards.append(reward)
        infos.append(info)
        if per_stage_eval is not None and (done or env.stage != stage_before):
            evals.append({"stage": stage_before, **per_stage_eval(env, stage_before)})
    return {
        "policy": policy.name,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "total_admit": int(sum(i["n_admit"] for i in infos)),
        "total_evict": int(sum(i["n_evict"] for i in infos)),
        "evals": evals,
        "rewards": rewards,
    }
