"""Full-memory teacher table for the label-free ι reward.

ι measures how much a capacity-M streaming bank B inflates the anomaly score of
held-out *normal* images relative to a teacher F that remembers everything:
F_t(n) = every stream patch of images 0..t except those in image n's
near-duplicate cluster. The teacher depends only on the cached stream (never
on a policy), so it is computed once per (cache, k) and looked up as a table:

    S_F[n, t] = max_p mean(k smallest squared L2 distances of patch p of image
                n to F_t(n)),   t >= n   (NaN for t < n)

The scorer is exactly evaluate.py's (mean of the k smallest SQUARED L2
distances, max over patches, no density reweighting): with singleton clusters
every bank the env can build is a subset of F_t(n) for every held-out n, so
S_B(n) >= S_F(n; t) and ι >= 0 up to float noise.
"""
import time
from typing import Optional

import numpy as np
import torch


def sq_l2(a: torch.Tensor, b: torch.Tensor,
          a2: Optional[torch.Tensor] = None,
          b2: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Squared L2 distance matrix [len(a), len(b)] in fp32 (one addmm).

    Shared by the teacher build and the env's S_B tracker so both sides of the
    ι log-ratio go through the same arithmetic. ``a2``/``b2`` are optional
    precomputed squared row norms.
    """
    if a2 is None:
        a2 = (a * a).sum(dim=1)
    if b2 is None:
        b2 = (b * b).sum(dim=1)
    d = torch.addmm(b2.unsqueeze(0), a, b.t(), beta=1.0, alpha=-2.0)
    d.add_(a2.unsqueeze(1))
    return d.clamp_min_(0.0)


def image_scores_from_topk(topd: torch.Tensor, n_images: int) -> torch.Tensor:
    """[n_images*P, k] k-smallest squared distances -> [n_images] image scores.

    Mean over the FINITE entries only (fewer than k candidates -> mean of what
    exists, matching evaluate.py's kk = min(k, |bank|)); a patch with no
    candidate at all yields NaN, which propagates to the image score.
    """
    finite = torch.isfinite(topd)
    cnt = finite.sum(dim=1)
    s = torch.where(finite, topd, torch.zeros_like(topd)).sum(dim=1)
    patch = s / cnt.clamp_min(1)
    patch = torch.where(cnt > 0, patch, torch.full_like(patch, float("nan")))
    return patch.view(n_images, -1).max(dim=1).values


# Distance-matrix budget per chunk (fp32 entries): 256M = 1GB. Larger chunks
# amortize the per-image top-k merges (the build's Python-overhead floor).
_TEACHER_CHUNK_ENTRIES = 256_000_000


def build_teacher(reader, k: int, clusters: Optional[np.ndarray] = None,
                  device: Optional[torch.device] = None, block: int = 32,
                  verbose: bool = True) -> dict:
    """Compute the teacher table S_F for ``reader``'s stream.

    Blocked over the stream (``block`` new images at a time): each block is
    one chunked squared-distance pass of ALL N*P patches (kept on ``device``)
    against the block's patches, reduced to a per-(query patch, block image)
    top-k, then merged image-by-image into a running per-patch top-k so
    S_F[:, t] is read off after every image t. Query chunks hold whole images,
    so image scores never leave the device until the [chunk, block] table.

    ``clusters`` (int [N]): near-duplicate cluster id per image; patches of
    n's cluster never enter n's teacher. None = singletons (own image only).
    Returns {"S_F" [N, N] float32, "clusters" int64 [N], "eps", "k", "n_images"}.
    """
    from patchcore.streaming.bank import knn_device

    device = device or knn_device()
    n, dim = reader.n_images, reader.dim
    clusters = np.arange(n, dtype=np.int64) if clusters is None \
        else np.asarray(clusters, dtype=np.int64)
    if len(clusters) != n:
        raise ValueError(f"clusters has {len(clusters)} entries, stream has {n}")
    emb = getattr(reader, "embeddings", None)
    if emb is not None:
        x_np = np.ascontiguousarray(emb, dtype=np.float32)
    else:
        x_np = np.stack([reader.image_patches(i) for i in range(n)]).astype(np.float32)
    p = x_np.shape[1]
    x = torch.from_numpy(x_np.reshape(n * p, dim)).to(device)
    del x_np
    x2 = (x * x).sum(dim=1)
    cl = torch.from_numpy(clusters).to(device)
    kk = min(k, p)

    block = max(1, min(block, n))
    ipc = max(1, _TEACHER_CHUNK_ENTRIES // max(block * p * p, 1))  # images/chunk
    run = torch.full((n * p, k), float("inf"), device=device)
    S_F = np.full((n, n), np.nan, dtype=np.float32)
    t_start = time.perf_counter()
    for b0 in range(0, n, block):
        b1 = min(b0 + block, n)
        nb = b1 - b0
        xb, xb2 = x[b0 * p:b1 * p], x2[b0 * p:b1 * p]
        cb = cl[b0:b1]
        for q0 in range(0, n, ipc):
            q1 = min(q0 + ipc, n)
            rows = slice(q0 * p, q1 * p)
            d = sq_l2(x[rows], xb, x2[rows], xb2).view((q1 - q0) * p, nb, p)
            blk = torch.topk(d, kk, dim=2, largest=False).values  # [Q, nb, kk]
            del d
            same = (cl[q0:q1].unsqueeze(1) == cb.unsqueeze(0))     # [ipc, nb]
            same = same.repeat_interleave(p, dim=0)                 # [Q, nb]
            blk.masked_fill_(same.unsqueeze(2), float("inf"))
            r = run[rows]
            scores = []
            for j in range(nb):
                r = torch.topk(torch.cat([r, blk[:, j]], dim=1), k,
                               dim=1, largest=False).values
                scores.append(image_scores_from_topk(r, q1 - q0))
            run[rows] = r
            S_F[q0:q1, b0:b1] = torch.stack(scores, dim=1).cpu().numpy()
        if verbose:
            el = time.perf_counter() - t_start
            pairs = b1 * p * n * p
            print(f"[teacher] {b1}/{n} images  {el:.1f}s  "
                  f"{pairs / max(el, 1e-9) / 1e9:.2f} Gpair/s", flush=True)
    # t < n is undefined (n not yet streamed) by contract.
    S_F[np.tril_indices(n, k=-1)] = np.nan
    final = S_F[:, n - 1]
    med = float(np.nanmedian(final)) if np.isfinite(final).any() else 1.0
    return {"S_F": S_F, "clusters": clusters, "eps": 0.01 * med,
            "k": int(k), "n_images": int(n)}


def save_teacher(path: str, teacher: dict) -> None:
    np.savez(path, S_F=teacher["S_F"], clusters=teacher["clusters"],
             eps=np.float64(teacher["eps"]), k=np.int64(teacher["k"]),
             n_images=np.int64(teacher["n_images"]))


def load_teacher(path: str) -> dict:
    with np.load(path) as z:
        return {"S_F": z["S_F"].astype(np.float32), "clusters": z["clusters"].astype(np.int64),
                "eps": float(z["eps"]), "k": int(z["k"]), "n_images": int(z["n_images"])}


def dhash_clusters(paths, max_hamming: int = 2) -> np.ndarray:
    """Near-duplicate clusters: 64-bit dHash (9x8 grayscale), hamming <= 2,
    connected components. Returns an int64 cluster id per path."""
    import PIL.Image
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    hashes = np.empty((len(paths), 8), dtype=np.uint8)
    for i, path in enumerate(paths):
        g = np.asarray(PIL.Image.open(path).convert("L").resize(
            (9, 8), PIL.Image.LANCZOS), dtype=np.int16)
        hashes[i] = np.packbits((g[:, 1:] > g[:, :-1]).reshape(-1))
    rows, cols = [], []
    for i in range(len(paths)):  # O(N^2) popcounts, vectorized per row
        ham = np.unpackbits(hashes[i] ^ hashes[i + 1:], axis=1).sum(axis=1)
        j = np.flatnonzero(ham <= max_hamming) + i + 1
        rows.extend([i] * len(j))
        cols.extend(j.tolist())
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(paths),) * 2)
    _, labels = connected_components(graph, directed=False)
    return labels.astype(np.int64)
