"""Dynamic, budgeted memory bank with add / evict / k-NN under drift.

The bank keeps a pure-numpy array as the source of truth. k-NN queries run as
exact brute-force L2 through torch on the fastest available device (CUDA/MPS
when present, multi-threaded CPU otherwise): at streaming scale (M ~ 2000) one
distance matrix per query batch is far cheaper than rebuilding a FAISS index
after every mutation, and it keeps the hot path off the 4-thread FAISS/OpenMP
pin. Final evaluation still goes through
:class:`patchcore.common.NearestNeighbourScorer` via :meth:`install_into`, so
anomaly scores stay byte-identical to stock PatchCore.

Expensive per-step derived quantities (member NN-2 redundancy, eviction
entry features) are cached per mutation step: the observation, the action
decoder, and the reward all consume the same bank state within a step, so
they share one computation instead of re-running the k-NN each time.
"""
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


# Bound each [chunk, M] distance matrix to ~64M float32 entries (256MB), so
# query batches stay memory-safe as the bank capacity M grows.
_KNN_CHUNK_ENTRIES = 64_000_000


def knn_device() -> torch.device:
    """Device used for brute-force k-NN (override with STREAMING_KNN_DEVICE).

    Auto-selects CUDA when available, otherwise multi-threaded CPU. MPS is
    deliberately not auto-selected: its per-call dispatch/sync overhead makes
    it slower than CPU at this bank size (set STREAMING_KNN_DEVICE=mps to try).
    """
    name = os.environ.get("STREAMING_KNN_DEVICE")
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def intra_batch_nn2(x: np.ndarray, device: Optional[torch.device] = None) -> np.ndarray:
    """Distance of each row of ``x`` to its nearest *other* row ([n] float32)."""
    n = len(x)
    if n < 2:
        return np.zeros(n, dtype=np.float32)
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(
        device or knn_device()
    )
    d = torch.cdist(t, t)
    d.fill_diagonal_(float("inf"))
    return d.min(dim=1).values.cpu().numpy().astype(np.float32)


# Per-slot metadata columns kept alongside the stored vectors. These feed both
# the observation vector and the per-entry eviction utility features.
@dataclass
class BankSnapshot:
    store: np.ndarray
    active: np.ndarray
    insert_step: np.ndarray
    insert_stage: np.ndarray
    hit_count: np.ndarray
    last_hit_step: np.ndarray
    size: int
    step: int


class DynamicMemoryBank:
    """A fixed-capacity memory bank of patch embeddings.

    Args:
        capacity: hard budget ``M`` (maximum number of active entries).
        dim: embedding dimension ``D``.
        nn_method: unused; accepted for backwards compatibility with the old
            FAISS-backed implementation.
        rebuild_every: unused; accepted for backwards compatibility.
        seed: RNG seed for redundancy subsampling.
        device: torch device for k-NN (defaults to :func:`knn_device`).
    """

    def __init__(
        self,
        capacity: int,
        dim: int,
        nn_method=None,
        rebuild_every: int = 1,
        seed: int = 0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.capacity = int(capacity)
        self.dim = int(dim)
        self._device = device or knn_device()
        self._rng = np.random.default_rng(seed)

        self._store = np.zeros((self.capacity, self.dim), dtype=np.float32)
        self._active = np.zeros(self.capacity, dtype=bool)
        self.insert_step = np.zeros(self.capacity, dtype=np.int64)
        self.insert_stage = np.zeros(self.capacity, dtype=np.int64)
        self.hit_count = np.zeros(self.capacity, dtype=np.int64)
        self.last_hit_step = np.zeros(self.capacity, dtype=np.int64)

        self._size = 0
        self._step = 0                 # logical mutation clock
        self._base_dirty = True
        self._base: Optional[torch.Tensor] = None  # active vectors on device
        # Maps a compact index position (0..size-1, in active-slot order) back to
        # the underlying slot id; rebuilt with the device tensor.
        self._index_to_slot = np.empty(0, dtype=np.int64)
        # Per-mutation-step caches of derived per-member quantities.
        self._nn2_cache: Optional[np.ndarray] = None
        self._nn2_step = -1
        self._ent_cache: Optional[np.ndarray] = None
        self._ent_step = -1

    # ---- basic state -----------------------------------------------------
    def __len__(self) -> int:
        return self._size

    @property
    def occupancy(self) -> float:
        return self._size / self.capacity if self.capacity else 0.0

    @property
    def step(self) -> int:
        return self._step

    def active_slots(self) -> np.ndarray:
        return np.flatnonzero(self._active)

    def vectors(self) -> np.ndarray:
        """Contiguous copy of the active vectors, in active-slot order."""
        return np.ascontiguousarray(self._store[self._active])

    # ---- mutation --------------------------------------------------------
    def _free_slots(self, n: int) -> np.ndarray:
        free = np.flatnonzero(~self._active)
        if len(free) < n:
            raise ValueError(
                f"cannot add {n} entries: only {len(free)} free slots "
                f"(size={self._size}, capacity={self.capacity})"
            )
        return free[:n]

    def add(self, vectors: np.ndarray, stage: int = 0) -> np.ndarray:
        """Insert ``vectors`` ([n, D]); returns the assigned slot ids.

        Raises if there is not enough free capacity — callers evict first.
        """
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None]
        n = len(vectors)
        if n == 0:
            return np.empty(0, dtype=np.int64)
        slots = self._free_slots(n)
        self._store[slots] = vectors
        self._active[slots] = True
        self.insert_step[slots] = self._step
        self.insert_stage[slots] = stage
        self.hit_count[slots] = 0
        self.last_hit_step[slots] = self._step
        self._size += n
        self._mark_mutated()
        return slots

    def evict(self, slot_ids: np.ndarray) -> None:
        """Deactivate the given slot ids (freeing them for reuse by ``add``)."""
        slot_ids = np.atleast_1d(np.asarray(slot_ids, dtype=np.int64))
        slot_ids = slot_ids[self._active[slot_ids]]
        if len(slot_ids) == 0:
            return
        self._active[slot_ids] = False
        self._size -= len(slot_ids)
        self._mark_mutated()

    def _mark_mutated(self) -> None:
        self._step += 1
        self._base_dirty = True

    # ---- indexing / queries ---------------------------------------------
    def _ensure_base(self) -> None:
        if not self._base_dirty:
            return
        slots = self.active_slots()
        self._index_to_slot = slots
        if len(slots) == 0:
            self._base = None
        else:
            self._base = torch.from_numpy(
                np.ascontiguousarray(self._store[slots])
            ).to(self._device)
        self._base_dirty = False

    def knn(
        self, queries: np.ndarray, k: int = 1, record_hits: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """k nearest neighbours of ``queries`` ([n, D]) among active entries.

        Returns ``(distances [n, k], slot_ids [n, k])``. Missing neighbours
        (fewer than ``k`` active entries) are returned as ``inf`` distance and
        ``-1`` slot id. When ``record_hits`` is set, the nearest slot's hit
        statistics are updated (used by eviction utility features).
        """
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[None]
        n = len(queries)
        if n == 0:
            return (
                np.zeros((0, k), dtype=np.float32),
                np.zeros((0, k), dtype=np.int64),
            )
        if self._size == 0:
            return (
                np.full((n, k), np.inf, dtype=np.float32),
                np.full((n, k), -1, dtype=np.int64),
            )
        self._ensure_base()
        kk = min(k, self._size)
        chunk = max(256, _KNN_CHUNK_ENTRIES // self._size)
        dist_chunks, idx_chunks = [], []
        for start in range(0, n, chunk):
            q = torch.from_numpy(queries[start:start + chunk]).to(self._device)
            d = torch.cdist(q, self._base)
            dd, ii = torch.topk(d, kk, dim=1, largest=False)
            dist_chunks.append(dd.cpu().numpy())
            idx_chunks.append(ii.cpu().numpy())
        dists = np.concatenate(dist_chunks, axis=0)
        idxs = np.concatenate(idx_chunks, axis=0)
        slots = self._index_to_slot[idxs]
        if record_hits and kk >= 1:
            nearest = slots[:, 0]
            np.add.at(self.hit_count, nearest, 1)
            self.last_hit_step[nearest] = self._step
            self._ent_step = -1  # hit stats feed entry_features; invalidate
        if kk < k:  # pad to requested width
            pad_d = np.full((n, k - kk), np.inf, dtype=np.float32)
            pad_i = np.full((n, k - kk), -1, dtype=np.int64)
            dists = np.concatenate([dists, pad_d], axis=1)
            slots = np.concatenate([slots, pad_i], axis=1)
        return dists.astype(np.float32), slots.astype(np.int64)

    def projected_mean(self, proj: np.ndarray) -> np.ndarray:
        """Mean of the active vectors projected through ``proj`` ([D, p] -> [p]).

        Runs on the cached device tensor, so it stays cheap at large M
        (the numpy equivalent ``vectors() @ proj`` copies and re-projects the
        whole bank every step).
        """
        if self._size == 0:
            return np.zeros(proj.shape[1], dtype=np.float32)
        self._ensure_base()
        p = torch.from_numpy(np.ascontiguousarray(proj, dtype=np.float32)).to(
            self._device
        )
        # mean-then-project == project-then-mean (linearity), but O(D*p) cheaper
        return (self._base.mean(dim=0) @ p).cpu().numpy().astype(np.float32)

    def _member_nn2(self) -> np.ndarray:
        """NN-2 distance of every active member to the rest, in active-slot
        order; cached per mutation step (observation, eviction features and
        reward all read it within one step)."""
        if self._nn2_step == self._step and self._nn2_cache is not None:
            return self._nn2_cache
        if self._size < 2:
            nn2 = np.zeros(0, dtype=np.float32)
        else:
            # k=2: the nearest is the member itself (distance 0); take the second.
            dists, _ = self.knn(self._store[self.active_slots()], k=2)
            nn2 = dists[:, 1]
        self._nn2_cache = nn2
        self._nn2_step = self._step
        return nn2

    def member_redundancy(self, sample: int = 2048) -> np.ndarray:
        """NN-2 distance of (up to ``sample``) active members to the rest.

        Small values mean the bank contains near-duplicates — budget wasted on
        redundancy. Returns one distance per sampled member. When a full
        per-member cache already exists for this step it is subsampled;
        otherwise only ``sample`` members are queried (O(sample * M) instead of
        the O(M^2) full computation, which matters for large banks).
        """
        if self._size < 2:
            return np.zeros(0, dtype=np.float32)
        if self._nn2_step == self._step and self._nn2_cache is not None:
            nn2 = self._nn2_cache
            if len(nn2) > sample:
                nn2 = self._rng.choice(nn2, size=sample, replace=False)
            return nn2
        if sample >= self._size:
            return self._member_nn2()
        slots = self._rng.choice(self.active_slots(), size=sample, replace=False)
        dists, _ = self.knn(self._store[slots], k=2)
        return dists[:, 1]

    def entry_features(self) -> np.ndarray:
        """Per-active-entry features ``[m, 4]`` for the eviction utility.

        Columns: normalized age, negative hit rate, negative NN-2 redundancy,
        distance to the recent-insertion centroid. All are computed so that a
        higher weighted sum means "more worth keeping". Cached per mutation
        step.
        """
        if self._ent_step == self._step and self._ent_cache is not None:
            return self._ent_cache
        slots = self.active_slots()
        m = len(slots)
        if m == 0:
            return np.zeros((0, 4), dtype=np.float32)
        age = (self._step - self.insert_step[slots]).astype(np.float32)
        age = age / (age.max() + 1e-6)
        exposure = np.maximum(self._step - self.insert_step[slots], 1)
        hit_rate = self.hit_count[slots] / exposure
        hit_rate = hit_rate / (hit_rate.max() + 1e-6)
        red = self._member_nn2() if m >= 2 else np.zeros(m, np.float32)
        red_n = red / (red.max() + 1e-6)
        centroid = self._store[slots[self.insert_step[slots] >= np.median(
            self.insert_step[slots])]].mean(axis=0, keepdims=True) if m else 0.0
        dist_recent = np.linalg.norm(self._store[slots] - centroid, axis=1)
        dist_recent = dist_recent / (dist_recent.max() + 1e-6)
        feats = np.stack([age, -hit_rate, -red_n, dist_recent], axis=1).astype(
            np.float32
        )
        self._ent_cache = feats
        self._ent_step = self._step
        return feats

    # ---- persistence -----------------------------------------------------
    def snapshot(self) -> BankSnapshot:
        return BankSnapshot(
            store=self._store.copy(),
            active=self._active.copy(),
            insert_step=self.insert_step.copy(),
            insert_stage=self.insert_stage.copy(),
            hit_count=self.hit_count.copy(),
            last_hit_step=self.last_hit_step.copy(),
            size=self._size,
            step=self._step,
        )

    def restore(self, snap: BankSnapshot) -> None:
        self._store = snap.store.copy()
        self._active = snap.active.copy()
        self.insert_step = snap.insert_step.copy()
        self.insert_stage = snap.insert_stage.copy()
        self.hit_count = snap.hit_count.copy()
        self.last_hit_step = snap.last_hit_step.copy()
        self._size = snap.size
        self._step = snap.step
        self._base_dirty = True
        self._nn2_step = -1
        self._ent_step = -1

    def install_into(self, scorer):
        """Load the active vectors into a stock NearestNeighbourScorer.

        This is the bridge to identical-to-PatchCore evaluation: the scorer's
        ``fit`` builds the same FAISS index and ``predict`` yields the same
        k-NN L2 anomaly scores.
        """
        scorer.fit(detection_features=[self.vectors()])
        return scorer

    @classmethod
    def from_vectors(
        cls,
        vectors: np.ndarray,
        capacity: int,
        stage: int = 0,
        nn_method=None,
        seed: int = 0,
    ) -> "DynamicMemoryBank":
        """Construct a bank pre-filled with ``vectors`` (e.g. a stage-0 coreset)."""
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        capacity = max(capacity, len(vectors))
        bank = cls(capacity, vectors.shape[1], seed=seed)
        bank.add(vectors, stage=stage)
        bank._step = 0  # initial fill is not counted as a mutation step
        bank.insert_step[:] = 0
        return bank
