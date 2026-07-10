"""Dynamic, budgeted memory bank with add / evict / k-NN under drift.

The bank keeps a pure-numpy array as the source of truth and rebuilds an exact
``faiss.IndexFlatL2`` after mutations (see the plan's FAISS decision). Rebuild
from numpy guarantees that evaluation is byte-identical to stock PatchCore: the
same vectors, installed through :class:`patchcore.common.NearestNeighbourScorer`,
produce the same k-NN L2 scores.
"""
import copy
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

import patchcore.common


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
        nn_method: a :class:`patchcore.common.FaissNN` used for k-NN. A fresh
            index is (re)built from the active vectors after each mutation
            (batched by ``rebuild_every``).
        rebuild_every: rebuild the FAISS index only every N mutations; the index
            is always rebuilt lazily before a query if stale.
        seed: RNG seed for redundancy subsampling.
    """

    def __init__(
        self,
        capacity: int,
        dim: int,
        nn_method: Optional[patchcore.common.FaissNN] = None,
        rebuild_every: int = 1,
        seed: int = 0,
    ) -> None:
        self.capacity = int(capacity)
        self.dim = int(dim)
        self.nn_method = nn_method or patchcore.common.FaissNN(False, 4)
        self.rebuild_every = int(rebuild_every)
        self._rng = np.random.default_rng(seed)

        self._store = np.zeros((self.capacity, self.dim), dtype=np.float32)
        self._active = np.zeros(self.capacity, dtype=bool)
        self.insert_step = np.zeros(self.capacity, dtype=np.int64)
        self.insert_stage = np.zeros(self.capacity, dtype=np.int64)
        self.hit_count = np.zeros(self.capacity, dtype=np.int64)
        self.last_hit_step = np.zeros(self.capacity, dtype=np.int64)

        self._size = 0
        self._step = 0                 # logical mutation clock
        self._mutations_since_build = 0
        self._index_dirty = True
        # Maps a compact index position (0..size-1, in active-slot order) back to
        # the underlying slot id; rebuilt with the FAISS index.
        self._index_to_slot = np.empty(0, dtype=np.int64)

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
        self._mutations_since_build += 1
        self._index_dirty = True

    # ---- indexing / queries ---------------------------------------------
    def _ensure_index(self, force: bool = False) -> None:
        if not self._index_dirty:
            return
        if (
            not force
            and self._mutations_since_build < self.rebuild_every
            and self.nn_method.search_index is not None
        ):
            return
        slots = self.active_slots()
        self._index_to_slot = slots
        if len(slots) == 0:
            self.nn_method.reset_index()
        else:
            self.nn_method.fit(np.ascontiguousarray(self._store[slots]))
        self._index_dirty = False
        self._mutations_since_build = 0

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
        if self._size == 0:
            return (
                np.full((n, k), np.inf, dtype=np.float32),
                np.full((n, k), -1, dtype=np.int64),
            )
        self._ensure_index()
        kk = min(k, self._size)
        # FaissNN.run returns (distances, indices) into the compact index order.
        dists, idxs = self.nn_method.run(kk, queries)
        dists = np.sqrt(np.clip(dists, 0, None))  # IndexFlatL2 returns squared L2
        slots = np.where(idxs >= 0, self._index_to_slot[idxs.clip(min=0)], -1)
        if record_hits and kk >= 1:
            nearest = slots[:, 0]
            valid = nearest >= 0
            np.add.at(self.hit_count, nearest[valid], 1)
            self.last_hit_step[nearest[valid]] = self._step
        if kk < k:  # pad to requested width
            pad_d = np.full((n, k - kk), np.inf, dtype=np.float32)
            pad_i = np.full((n, k - kk), -1, dtype=np.int64)
            dists = np.concatenate([dists, pad_d], axis=1)
            slots = np.concatenate([slots, pad_i], axis=1)
        return dists.astype(np.float32), slots.astype(np.int64)

    def member_redundancy(self, sample: int = 2048) -> np.ndarray:
        """NN-2 distance of (up to ``sample``) active members to the rest.

        Small values mean the bank contains near-duplicates — budget wasted on
        redundancy. Returns one distance per sampled member.
        """
        if self._size < 2:
            return np.zeros(0, dtype=np.float32)
        slots = self.active_slots()
        if len(slots) > sample:
            slots = self._rng.choice(slots, size=sample, replace=False)
        queries = self._store[slots]
        # k=2: the nearest is the member itself (distance 0); take the second.
        dists, _ = self.knn(queries, k=2)
        return dists[:, 1]

    def entry_features(self) -> np.ndarray:
        """Per-active-entry features ``[m, 4]`` for the eviction utility.

        Columns: normalized age, negative hit rate, negative NN-2 redundancy,
        distance to the recent-insertion centroid. All are computed so that a
        higher weighted sum means "more worth keeping".
        """
        slots = self.active_slots()
        m = len(slots)
        if m == 0:
            return np.zeros((0, 4), dtype=np.float32)
        age = (self._step - self.insert_step[slots]).astype(np.float32)
        age = age / (age.max() + 1e-6)
        exposure = np.maximum(self._step - self.insert_step[slots], 1)
        hit_rate = self.hit_count[slots] / exposure
        hit_rate = hit_rate / (hit_rate.max() + 1e-6)
        red = self.member_redundancy(sample=m) if m >= 2 else np.zeros(m, np.float32)
        if len(red) != m:  # member_redundancy subsampled; recompute full
            dists, _ = self.knn(self._store[slots], k=2)
            red = dists[:, 1]
        red_n = red / (red.max() + 1e-6)
        centroid = self._store[slots[self.insert_step[slots] >= np.median(
            self.insert_step[slots])]].mean(axis=0, keepdims=True) if m else 0.0
        dist_recent = np.linalg.norm(self._store[slots] - centroid, axis=1)
        dist_recent = dist_recent / (dist_recent.max() + 1e-6)
        return np.stack([age, -hit_rate, -red_n, dist_recent], axis=1).astype(
            np.float32
        )

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
        self._index_dirty = True
        self._mutations_since_build = self.rebuild_every  # force rebuild next query

    def install_into(
        self, scorer: patchcore.common.NearestNeighbourScorer
    ) -> patchcore.common.NearestNeighbourScorer:
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
        nn_method: Optional[patchcore.common.FaissNN] = None,
        seed: int = 0,
    ) -> "DynamicMemoryBank":
        """Construct a bank pre-filled with ``vectors`` (e.g. a stage-0 coreset)."""
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        capacity = max(capacity, len(vectors))
        bank = cls(capacity, vectors.shape[1], nn_method=nn_method, seed=seed)
        bank.add(vectors, stage=stage)
        bank._step = 0  # initial fill is not counted as a mutation step
        bank.insert_step[:] = 0
        return bank
