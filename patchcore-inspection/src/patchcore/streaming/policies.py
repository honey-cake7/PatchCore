"""Maintenance policies: hand-designed baselines and the learned PPO wrapper.

Every policy drives the same environment transition through
:meth:`MemoryMaintenanceEnv.step_with_decision`, so all policies share budget,
reward logging, and evaluation hooks — only the admit/evict decision differs.
The common interface is ``step(env) -> (reward, done, info)``.
"""
import abc
import contextlib
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

    def __init__(self, period: int = 50, device: str = None, seed: int = 0):
        self.period = period
        self.device = device  # None -> auto (GPU when available)
        self.rng = np.random.default_rng(seed)
        self._buffer = []
        self._count = 0

    def step(self, env):
        import torch

        import patchcore.sampler
        from patchcore.streaming.bank import knn_device

        self._buffer.append(env.current_batch.copy())
        self._count += 1
        if self._count % self.period == 0:
            pool = np.concatenate([env.bank.vectors()] + self._buffer, axis=0)
            if len(pool) > env.capacity:
                pct = env.capacity / len(pool)
                device = torch.device(self.device) if self.device else knn_device()
                sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(
                    percentage=pct, device=device
                )
                from patchcore.streaming.bank import pinned_global_seed

                with pinned_global_seed(int(self.rng.integers(2**31))):
                    pool = sampler.run(np.ascontiguousarray(pool, dtype=np.float32))
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

        # Reuse the obs the env already computed for this state; re-observing
        # would double-update the RunningNorm warmup relative to training.
        obs = env.last_obs
        if obs is None:
            obs = env._observe(update=False)
        with torch.no_grad():
            t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            dist, _ = self.ac(t)
            action = dist.mean.squeeze(0).cpu().numpy()
        return env.decode_action(action)


# ---- label-free / geometric baselines for the ι benchmark ------------------
def _admissible_dev(env):
    """The current admissible batch as a k-NN-device tensor (the env keeps a
    device gather; fall back to one upload for envs that don't)."""
    import torch

    t = getattr(env, "_admissible_dev", None)
    if t is None:
        t = torch.from_numpy(
            np.ascontiguousarray(env.current_batch, dtype=np.float32)
        ).to(env.bank.device)
    return t


# Bound each [rows, n] distance / adjacency block, mirroring bank.knn.
_NET_CHUNK_ENTRIES = 64_000_000


def greedy_net(x, r: float):
    """Greedy r-net of the rows of ``x`` in row (= priority) order, on device.

    Returns a bool keep-mask equal to the sequential rule "walk the rows in
    order, keep a row iff it is > r from every row kept so far" — computed
    without a Python loop over rows. The r-graph (pairwise L2 <= r) is built
    once in chunked cdist blocks, then the lexicographically-first maximal
    independent set is peeled in parallel rounds: a row joins once no
    undecided lower-index neighbour remains, and the neighbours of joiners
    drop out. That fixed point IS the sequential greedy result. Rounds are
    bounded by the longest priority-decreasing chain in the r-graph — small
    for spatially unordered priorities. Ceiling: the dense bool adjacency is
    n^2 bytes (~0.4 GB at n=20k); a sparse edge list is the upgrade if M grows
    past ~50k.
    """
    import torch

    n = len(x)
    dev = x.device
    if n == 0:
        return torch.zeros(0, dtype=torch.bool, device=dev)
    chunk = max(1, _NET_CHUNK_ENTRIES // n)
    adj = torch.empty((n, n), dtype=torch.bool, device=dev)
    for s in range(0, n, chunk):
        adj[s:s + chunk] = torch.cdist(x[s:s + chunk], x) <= r
    adj.fill_diagonal_(False)

    keep = torch.zeros(n, dtype=torch.bool, device=dev)
    undecided = torch.ones(n, dtype=torch.bool, device=dev)
    while bool(undecided.any()):
        blocked = torch.empty(n, dtype=torch.bool, device=dev)
        for s in range(0, n, chunk):
            blk = adj[s:s + chunk] & undecided.unsqueeze(0)
            # strictly-lower columns only: j < s + local_row
            blocked[s:s + chunk] = torch.tril(blk, diagonal=s - 1).any(dim=1)
        join = undecided & ~blocked
        keep |= join
        covered = torch.empty(n, dtype=torch.bool, device=dev)
        for s in range(0, n, chunk):
            covered[s:s + chunk] = (adj[s:s + chunk] & join.unsqueeze(0)).any(dim=1)
        undecided &= ~(join | covered)
    return keep


def _min_positive_nn(x) -> float:
    """Smallest non-zero nearest-neighbour distance among the rows of ``x``."""
    import torch

    n = len(x)
    if n < 2:
        return 0.0
    chunk = max(1, _NET_CHUNK_ENTRIES // n)
    best = float("inf")
    for s in range(0, n, chunk):
        d = torch.cdist(x[s:s + chunk], x)
        d[d <= 0] = float("inf")  # self + exact duplicates
        best = min(best, float(d.min()))
    return best if np.isfinite(best) else 0.0


class KCenterPolicy(BasePolicy):
    """Streaming k-center (Charikar et al. 1997 doubling algorithm), patch level.

    Keeps the bank an r-net: an incoming patch is admitted iff it is > r from
    every bank member (and from the patches admitted with it, via the batch's
    greedy net). When admissions would overflow the capacity M, r doubles and
    bank ∪ admits is thinned to a greedy r-net — oldest members first, so
    centers persist — repeating until it fits; the thinned-out members are
    this step's evictions. The k-center objective is coverage radius, which is
    exactly what the max-over-patches PatchCore score is sensitive to, making
    this the principled geometric baseline next to the greedy coreset.

    r starts at half the bank's smallest positive member NN distance (the
    doubling algorithm's lower bound on OPT), recomputed at each episode start.
    """

    name = "kcenter"

    def __init__(self):
        self.r = None

    def decide(self, env):
        import torch

        empty = np.empty(0, dtype=np.int64)
        bank = env.bank
        if self.r is None or env._t == env._start:
            bank._ensure_base()
            slots = bank.active_slots()
            self.r = 0.5 * _min_positive_nn(
                bank._base[torch.from_numpy(slots).to(bank.device)])
        nn = env.batch_novelty
        if len(nn) == 0:
            return empty, empty
        # an empty bank reports novelty 0 (inf masked) — everything is novel
        cand = np.flatnonzero(nn > self.r) if len(bank) else np.arange(len(nn))
        if len(cand) == 0:
            return empty, empty
        cand = cand[np.argsort(-nn[cand], kind="stable")]  # most novel first
        A = _admissible_dev(env)
        x = A[torch.from_numpy(cand).to(A.device)]
        admit = cand[greedy_net(x, self.r).cpu().numpy()]
        if len(bank) + len(admit) <= env.capacity:
            return admit, empty

        # overflow: doubling step over bank ∪ admits
        bank._ensure_base()
        slots = bank.active_slots()
        slots = slots[np.argsort(bank.insert_step[slots], kind="stable")]
        pool = torch.cat([
            bank._base[torch.from_numpy(slots).to(bank.device)],
            A[torch.from_numpy(admit).to(A.device)].to(bank.device),
        ])
        if self.r <= 0:  # bank had no positive spacing yet
            self.r = 0.5 * _min_positive_nn(pool)
            if self.r <= 0:  # all identical: one center suffices
                self.r = 1e-6
        kept = torch.arange(len(pool), device=pool.device)
        while len(kept) > env.capacity:
            self.r *= 2.0
            kept = kept[greedy_net(pool[kept], self.r)]
        kept = kept.cpu().numpy()
        is_kept = np.zeros(len(pool), dtype=bool)
        is_kept[kept] = True
        n_bank = len(slots)
        return admit[is_kept[n_bank:]], slots[~is_kept[:n_bank]]


@contextlib.contextmanager
def trial_mutation(bank, evict_slots: np.ndarray, vectors: np.ndarray):
    """Temporarily apply evict-then-add to ``bank``; undo exactly on exit.

    Equivalent to ``snap = bank.snapshot(); mutate; ...; bank.restore(snap)``
    but O(touched rows) instead of O(M*D): restore() copies the whole host
    store and marks the device mirror dirty, i.e. a full [M, D] re-upload per
    trial — several per step for the lookahead policy. Here only the evicted
    rows and the free rows add() will fill are saved/restored, on host AND on
    the device mirror, plus the per-step caches and the bank RNG.
    """
    import torch

    bank._ensure_base()
    evict_slots = np.unique(np.asarray(evict_slots, dtype=np.int64))
    evict_slots = evict_slots[bank._active[evict_slots]]
    vectors = np.ascontiguousarray(vectors, dtype=np.float32).reshape(-1, bank.dim)
    # add() fills the lowest free slots after the eviction; those lie within
    # (currently free)[:n] ∪ evicted.
    free = np.flatnonzero(~bank._active)[: len(vectors)]
    touched = np.union1d(evict_slots, free)
    arrays = ("_store", "_active", "insert_step", "insert_stage",
              "hit_count", "last_hit_step")
    saved = {a: getattr(bank, a)[touched].copy() for a in arrays}
    tdev = torch.from_numpy(touched).to(bank.device)
    base_rows = bank._base[tdev].clone()
    act_rows = bank._active_dev[tdev].clone()
    scalars = ("_size", "_step", "_nn2_cache", "_nn2_step", "_ent_cache",
               "_ent_step", "_nn2_sample_cache", "_nn2_sample_step",
               "_nn2_slot", "_nn2_slot_step")
    saved_scalars = {a: getattr(bank, a) for a in scalars}
    rng_state = bank._rng.bit_generator.state
    try:
        bank.evict(evict_slots)
        if len(vectors):
            bank.add(vectors)
        yield bank
    finally:
        for a in arrays:
            getattr(bank, a)[touched] = saved[a]
        bank._base[tdev] = base_rows
        bank._active_dev[tdev] = act_rows
        for a in scalars:
            setattr(bank, a, saved_scalars[a])
        bank._rng.bit_generator.state = rng_state


class MyopicIotaPolicy(BasePolicy):
    """Label-free one-step lookahead on the held-out normal score.

    Candidates: no-op ∪ {admit the top-m most novel patches, m ∈ SIZES} ×
    {evict oldest, evict most redundant}. Each is applied to the bank as a
    trial, scored by the mean log S_B over the held-out images seen so far
    (``env.heldout_image_scores``), and undone; the argmin is taken. The
    teacher term of ι is constant across candidates at a fixed t, so it
    cancels: this needs only the env's held-out set (heldout_frac > 0), no
    teacher — i.e. it is deployable. Ties (an admission that moves no
    held-out image's max patch) resolve to the earlier candidate, so no-op
    wins them and the policy writes only when it measurably helps.
    """

    name = "myopic_iota"
    SIZES = (1, 4, 16, 64)
    _FLOOR = 1e-12  # log guard; S_B > 0 since held-out images are never admitted

    def __init__(self, exact_trials: bool = False):
        # exact_trials=True scores every candidate by mutating a trial bank and
        # running the env's full recompute — the reference the fast path is
        # tested against. Default: one shared k-NN query + per-candidate merges.
        self.exact_trials = exact_trials

    def decide(self, env):
        empty = np.empty(0, dtype=np.int64)
        if not hasattr(env, "heldout_seen_ids") or len(env.heldout_ids) == 0:
            # Without a held-out set there is nothing to look ahead on; act as
            # static rather than crash generic all-BASELINES loops (e.g.
            # train_ppo --eval_baselines). run_streaming_baseline refuses this
            # configuration up front.
            if not getattr(self, "_warned", False):
                print("[myopic_iota] WARNING: env has no held-out set "
                      "(heldout_frac=0) — acting as static")
                self._warned = True
            return empty, empty
        if len(env.heldout_seen_ids()) == 0 or len(env.batch_novelty) == 0:
            return empty, empty
        cands, order = self.candidates(env)
        bank = env.bank
        if self.exact_trials or not hasattr(env, "heldout_pool_seen"):
            A = env.current_batch
            per_cand = []
            for admit, ev in cands:
                with trial_mutation(bank, ev, A[admit]):
                    per_cand.append(env.heldout_image_scores(bank))
        else:
            per_cand = self._merged_scores(env, cands, order)
        best, best_score = 0, float("inf")
        for i, s in enumerate(per_cand):
            s = np.asarray(s, dtype=np.float64)
            score = float(np.mean(np.log(np.maximum(s, self._FLOOR))))
            if score < best_score:
                best, best_score = i, score
        return cands[best]

    def candidates(self, env):
        """The deduplicated candidate list [(admit_idx, evict_slots)] (no-op
        first) and the novelty order the admissions are prefixes of."""
        empty = np.empty(0, dtype=np.int64)
        bank = env.bank
        nn = env.batch_novelty
        order = np.argsort(-nn, kind="stable")
        free = env.capacity - len(bank)
        slots = bank.active_slots()
        red_key = None

        cands = [(empty, empty)]
        keys = {((), ())}
        for m in self.SIZES:
            admit = order[:m]
            n_ev = max(0, len(admit) - free)
            for kind in ("oldest", "redundant"):
                if kind == "oldest":
                    key = bank.insert_step[slots] if len(slots) else np.empty(0)
                else:
                    if red_key is None and n_ev > 0:
                        red_key = bank.entry_features()[:, 2]
                    key = red_key if red_key is not None else np.empty(0)
                ev = _evict_by_key(env, n_ev, key)
                k = (tuple(admit.tolist()), tuple(sorted(ev.tolist())))
                if k not in keys:  # e.g. both evict rules when nothing is evicted
                    keys.add(k)
                    cands.append((admit, ev))
        return cands, order

    def _merged_scores(self, env, cands, order):
        """S_B per seen held-out image for every candidate, from one query.

        Every candidate is "current bank minus <= E evicted slots plus a prefix
        of the top-64 novel patches". So: one top-k query of the held-out pool
        against the live bank; rows whose top-k holds a slot that ANY
        candidate evicts get a top-(k+E) re-query (dropping <= E ids still
        leaves the true top-k of the remainder inside it); distances to the
        novel patches are reduced once per admit prefix to their k smallest.
        Each candidate is then a mask + a 2k-wide top-k merge on device. Same
        squared-L2 arithmetic as the env's scorer, so it matches the
        trial-bank recompute up to float rounding of the new-vector columns
        (tested). Cost per step ~ one O(R x M) query instead of one per
        candidate.
        """
        import torch

        from patchcore.streaming.teacher import image_scores_from_topk, sq_l2

        bank = env.bank
        pool, pool2, n_seen, k = env.heldout_pool_seen()
        dev = pool.device
        inf = float("inf")
        evicted = [ev for _, ev in cands if len(ev)]
        E = max((len(ev) for ev in evicted), default=0)
        any_ev = torch.zeros(bank.capacity + 1, dtype=torch.bool, device=dev)
        if evicted:
            any_ev[torch.from_numpy(np.unique(np.concatenate(evicted))).to(dev)] = True

        # The distance blocks replicate env._bank_topk_sq (same norms, mask,
        # row chunking) so the bank-side values are bit-identical to the
        # env's full recompute; the top-k AND the affected rows' top-(k+E)
        # come from the same block (a separate re-query of a row subset
        # rounds differently).
        bank._ensure_base()
        base, inactive = bank._base, ~bank._active_dev
        b2 = (base * base).sum(dim=1)
        kk, ke = min(k, bank.capacity), min(k + E, bank.capacity)
        chunk = max(256, 64_000_000 // max(bank.capacity, 1))
        ds, rows_a, d_a, ids_a = [], [], [], []
        for r0 in range(0, len(pool), chunk):
            blk = sq_l2(pool[r0:r0 + chunk], base, pool2[r0:r0 + chunk], b2)
            blk.masked_fill_(inactive.unsqueeze(0), inf)
            v, i = torch.topk(blk, kk, dim=1, largest=False)
            ds.append(v)
            if evicted:
                i = i.masked_fill(~torch.isfinite(v), -1)  # -1 -> False entry
                ra = torch.nonzero(any_ev[i].any(dim=1)).squeeze(1)
                if len(ra):
                    v2, i2 = torch.topk(blk[ra], ke, dim=1, largest=False)
                    rows_a.append(ra + r0)
                    d_a.append(v2)
                    ids_a.append(i2.masked_fill(~torch.isfinite(v2), -1))
        d = torch.cat(ds)
        if kk < k:
            d = torch.cat([d, torch.full((len(d), k - kk), inf, device=dev)], 1)
        if rows_a:
            rows_a, d_a, ids_a = torch.cat(rows_a), torch.cat(d_a), torch.cat(ids_a)
        else:
            rows_a = None

        # k smallest new-vector distances per admit prefix (shared by both
        # eviction rules of the same m)
        top = order[: max(self.SIZES)]
        A = _admissible_dev(env).to(dev)
        d_new = sq_l2(pool, A[torch.from_numpy(top).to(dev)], pool2)
        new_k = {}
        for admit, _ in cands:
            m = len(admit)  # admit == top[:m] by construction
            if m and m not in new_k:
                new_k[m] = torch.topk(d_new[:, :m], min(k, m), dim=1,
                                      largest=False).values

        out = []
        for admit, ev in cands:
            dd = d
            if len(ev) and rows_a is not None:
                ev_mask = torch.zeros(bank.capacity + 1, dtype=torch.bool, device=dev)
                ev_mask[torch.from_numpy(ev).to(dev)] = True
                ext = d_a.masked_fill(ev_mask[ids_a], inf)
                dd = d.clone()
                v = torch.topk(ext, min(k, ext.shape[1]), dim=1, largest=False).values
                if v.shape[1] < k:
                    v = torch.cat([v, torch.full((len(v), k - v.shape[1]), inf,
                                                 device=dev)], 1)
                dd[rows_a] = v
            if len(admit):
                dd = torch.topk(torch.cat([dd, new_k[len(admit)]], dim=1), k,
                                dim=1, largest=False).values
            out.append(image_scores_from_topk(dd, n_seen))
        return list(torch.stack(out).cpu().numpy())


BASELINES = {
    p.name: p
    for p in [StaticPolicy, FIFOPolicy, RandomPolicy, ReservoirPolicy,
              StreamingGreedyCoresetPolicy, PeriodicCoresetPolicy,
              KCenterPolicy, MyopicIotaPolicy]
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
        "infos": infos,
    }
