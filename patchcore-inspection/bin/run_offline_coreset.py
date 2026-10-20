"""Offline PatchCore at an exact memory budget M — the reference rows for the
memory sweep and the drift experiment.

    offline_stage0  greedy coreset of ALL stage-0 (pre-shift) stream images,
                    frozen — "offline PatchCore built on phase 1"
    offline_all     greedy coreset of the WHOLE stream (all phases) —
                    "offline PatchCore rebuilt on everything" (upper bound)

Both are scored on every stage's labeled test set (the held-out part when
--calib_split is set) and APPENDED to <out>/results.csv, replacing any earlier
offline_* rows, so the per-class summary / harvester pick them up as policies.
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

FIELDS = ["policy", "seed", "stage", "image_auroc", "pixel_auroc", "pro"]


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
@click.option("--out", required=True, help="result dir holding results.csv")
def main(cache_dir, capacity, n_nn, seeds, calib_split, max_patches, out):
    stream = EmbeddingCacheReader(os.path.join(cache_dir, "stream"))
    test_dir = os.path.join(cache_dir, "test")
    tests = [EmbeddingCacheReader(os.path.join(test_dir, s))
             for s in sorted(d for d in os.listdir(test_dir) if d.startswith("stage_"))]
    tests = split_test_readers(tests, calib_split, "eval")
    patch_shape = tuple(stream.manifest.get("patch_shape", (0, 0))) or None
    imagesize = stream.manifest.get("imagesize")

    pools = {
        "offline_stage0": [i for i in range(stream.n_images) if stream.stage_of(i) == 0],
        "offline_all": list(range(stream.n_images)),
    }
    rows = []
    for name, ids in pools.items():
        feats = stream.flat_slice(ids)
        if max_patches and len(feats) > max_patches:
            keep = np.random.default_rng(0).choice(len(feats), max_patches, replace=False)
            feats = feats[np.sort(keep)]
        print(f"[{name}] pool {len(ids)} images = {len(feats)} patches -> M={capacity}")
        for seed in (int(s) for s in seeds.split(",")):
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
            old = [r for r in csv.DictReader(f) if not r["policy"].startswith("offline_")]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(old + rows)
    print(f"appended {len(rows)} offline rows -> {path}")


if __name__ == "__main__":
    main()
