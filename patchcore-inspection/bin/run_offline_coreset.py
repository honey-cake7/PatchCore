"""Offline PatchCore at an exact memory budget M — the reference rows for the
memory sweep and the drift experiment.

    offline_stage0  greedy coreset of ALL stage-0 (pre-shift) stream images,
                    frozen — "offline PatchCore built on phase 1"
    offline_all     greedy coreset of the WHOLE stream (all phases) —
                    "offline PatchCore rebuilt on everything" (upper bound)

Both are scored on every stage's labeled test set (the held-out part when
--calib_split is set) and APPENDED to <out>/results.csv, replacing any earlier
offline_* rows, so the per-class summary / harvester pick them up as policies.

--full_memory instead scores the UNBUDGETED banks (no coreset) — the
full-memory teacher the ι reward compares against, i.e. the "is less really
more?" check:

    full_stage0     every stage-0 stream patch
    full_all        every stream patch (Kvasir: ~1.9M x 1024 fp32 = 7.8 GB)

These are deterministic (no seed), so one row set is written under the first
--seeds value; they replace earlier full_* rows and leave offline_* rows alone.
"""
import csv
import os
import sys

import click
import torch  # noqa: F401 — load before faiss (macOS libomp)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from patchcore.streaming.cache import EmbeddingCacheReader, split_test_readers
from patchcore.streaming.evaluate import evaluate_bank_on_stage
from patchcore.streaming.experiments import coreset_bank

# same superset schema as run_streaming_baseline.py
FIELDS = ["policy", "seed", "train_seed", "stage", "image_auroc", "pixel_auroc",
          "pro", "writes", "evict", "iota"]


def chunked_patch_scores(bank: np.ndarray, flat: np.ndarray, k: int,
                         device=None, bank_chunk: int = 131072,
                         query_chunk: int = 4096) -> np.ndarray:
    """Mean of the k smallest squared L2 distances to ``bank``, per query row.

    Same scorer as evaluate._torch_patch_scores, but the bank is streamed to
    the device in ``bank_chunk`` rows too (that one uploads the whole bank and
    sizes query chunks by it — at 1.9M rows that is 7.8 GB resident plus a
    tiny query chunk). Queries stay resident; each bank chunk is uploaded once
    and its per-query top-k is merged exactly into the running top-k (top-k of
    a union = top-k of the per-part top-ks). Peak device memory ≈ bank chunk
    (0.5 GB) + queries + one [query_chunk, bank_chunk] fp32 block (2 GB).
    ``bank`` may be a memmap/numpy array; fp32 throughout.
    """
    import torch

    from patchcore.streaming.bank import knn_device

    device = device or knn_device()
    kk = min(k, len(bank))
    q = torch.from_numpy(np.ascontiguousarray(flat, dtype=np.float32)).to(device)
    best = torch.full((len(q), kk), float("inf"), device=device)
    for b0 in range(0, len(bank), bank_chunk):
        b = torch.from_numpy(np.ascontiguousarray(
            bank[b0:b0 + bank_chunk], dtype=np.float32)).to(device)
        kb = min(kk, len(b))
        for q0 in range(0, len(q), query_chunk):
            d = torch.cdist(q[q0:q0 + query_chunk], b) ** 2
            vals = torch.topk(d, kb, dim=1, largest=False).values
            merged = torch.cat([best[q0:q0 + query_chunk], vals], dim=1)
            best[q0:q0 + query_chunk] = torch.topk(
                merged, kk, dim=1, largest=False).values
        del b
    return best.mean(dim=1).cpu().numpy().astype(np.float32)


def evaluate_full_memory(bank: np.ndarray, reader, k: int, patch_shape,
                         imagesize, **chunk_kw) -> dict:
    """evaluate_bank_on_stage's metrics for a raw (unbudgeted) patch array.

    Mirrors evaluate.py's scoring tail (max-over-patches image score,
    RescaleSegmentor pixel maps, AUROC/PRO) — only the k-NN is swapped for the
    bank-chunked one; building a DynamicMemoryBank would copy the 7.8 GB array
    twice more.
    """
    import patchcore.common
    import patchcore.metrics
    from patchcore.patchcore import PatchMaker
    from patchcore.streaming.bank import knn_device

    emb = reader.embeddings
    n, p, d = emb.shape
    scores = chunked_patch_scores(bank, emb.reshape(n * p, d), k, **chunk_kw)
    scores = scores.reshape(n, p)
    image_scores = np.asarray(PatchMaker(3, stride=1).score(scores.copy()),
                              dtype=np.float64)
    labels = np.asarray(reader.labels, dtype=int)
    out = {"n_images": int(n), "n_bank": int(len(bank)),
           "image_auroc": float("nan")}
    if len(np.unique(labels)) > 1:
        out["image_auroc"] = float(patchcore.metrics.compute_imagewise_retrieval_metrics(
            image_scores, labels)["auroc"])
    if (patch_shape is not None and imagesize is not None
            and reader.masks is not None and patch_shape[0] * patch_shape[1] == p):
        segmentor = patchcore.common.RescaleSegmentor(
            device=knn_device(), target_size=(imagesize, imagesize))
        segs = np.stack(segmentor.convert_to_segmentation(
            scores.reshape(n, *patch_shape)))
        masks = np.asarray(reader.masks)
        out["pixel_auroc"] = float(patchcore.metrics.compute_pixelwise_retrieval_metrics(
            list(segs), list(masks))["auroc"])
        anom = [i for i in range(len(masks)) if masks[i].sum() > 0]
        if anom:
            out["pro"] = float(patchcore.metrics.compute_pro(
                [segs[i] for i in anom], [masks[i] for i in anom])["pro"])
    return out


@click.command()
@click.option("--cache_dir", required=True)
@click.option("--capacity", type=int, required=True)
@click.option("--n_nn", type=int, default=5)
@click.option("--seeds", default="0,1,2,3,4")
@click.option("--calib_split", type=float, default=0.0)
@click.option("--max_patches", type=int, default=0,
              help="Uniformly subsample the coreset pool to this many patches "
                   "before greedy selection (0 = all). Ceiling: the whole-stream "
                   "pool of a long endoscopy stream is ~2-3M x 1024 fp32 "
                   "(8-11 GB host + GPU); cap it only if that OOMs.")
@click.option("--full_memory", is_flag=True, default=False,
              help="Score the unbudgeted full_stage0 / full_all banks (every "
                   "stream patch, no coreset) INSTEAD of the offline coresets; "
                   "--capacity is then unused.")
@click.option("--out", required=True, help="result dir holding results.csv")
def main(cache_dir, capacity, n_nn, seeds, calib_split, max_patches, full_memory, out):
    stream = EmbeddingCacheReader(os.path.join(cache_dir, "stream"))
    test_dir = os.path.join(cache_dir, "test")
    tests = [EmbeddingCacheReader(os.path.join(test_dir, s))
             for s in sorted(d for d in os.listdir(test_dir) if d.startswith("stage_"))]
    tests = split_test_readers(tests, calib_split, "eval")
    patch_shape = tuple(stream.manifest.get("patch_shape", (0, 0))) or None
    imagesize = stream.manifest.get("imagesize")

    stage0_ids = [i for i in range(stream.n_images) if stream.stage_of(i) == 0]
    prefix = "full_" if full_memory else "offline_"
    pools = {
        prefix + "stage0": stage0_ids,
        prefix + "all": list(range(stream.n_images)),
    }
    seed_list = [int(s) for s in seeds.split(",")]
    rows = []
    for name, ids in pools.items():
        feats = stream.flat_slice(ids)
        if max_patches and len(feats) > max_patches:
            keep = np.random.default_rng(0).choice(len(feats), max_patches, replace=False)
            feats = feats[np.sort(keep)]
        if full_memory:
            print(f"[{name}] full memory: {len(ids)} images = {len(feats)} patches")
            for stage, reader in enumerate(tests):
                m = evaluate_full_memory(feats, reader, n_nn, patch_shape, imagesize)
                rows.append({"policy": name, "seed": seed_list[0], "stage": stage,
                             "image_auroc": m.get("image_auroc", float("nan")),
                             "pixel_auroc": m.get("pixel_auroc", float("nan")),
                             "pro": m.get("pro", float("nan"))})
            print(f"  image AUROC per stage: "
                  f"{[round(r['image_auroc'], 3) for r in rows if r['policy'] == name]}")
            del feats
            continue
        print(f"[{name}] pool {len(ids)} images = {len(feats)} patches -> M={capacity}")
        for seed in seed_list:
            bank = coreset_bank(feats, capacity, seed=seed)
            for stage, reader in enumerate(tests):
                m = evaluate_bank_on_stage(bank, reader, n_nn, patch_shape, imagesize)
                rows.append({"policy": name, "seed": seed, "stage": stage,
                             "image_auroc": m.get("image_auroc", float("nan")),
                             "pixel_auroc": m.get("pixel_auroc", float("nan")),
                             "pro": m.get("pro", float("nan"))})
            s0 = [r["image_auroc"] for r in rows if r["policy"] == name
                  and r["seed"] == seed]
            print(f"  seed={seed} image AUROC per stage: {[round(v, 3) for v in s0]}")
        del feats

    path = os.path.join(out, "results.csv")
    old = []
    if os.path.exists(path):
        with open(path) as f:
            old = [r for r in csv.DictReader(f) if not r["policy"].startswith(prefix)]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(old + rows)
    print(f"appended {len(rows)} {prefix}* rows -> {path}")


if __name__ == "__main__":
    main()
