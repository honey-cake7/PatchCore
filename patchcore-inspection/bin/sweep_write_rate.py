"""Does the ι reward have a slope in the write rate? (no training, no labels for ι)

Runs fixed-rate policies — k admitted patches per image, k in --rates — under
each content/eviction rule, every one scored on the SAME held-out set per seed,
and reports the anytime ι return (mean per-step -ι, the training objective)
plus, with --auroc, the labelled per-stage image AUROC and stage-0 forgetting.

    random: k uniformly random admissible patches, random eviction
    novel:  the k most novel patches, oldest evicted (fifo)

k=0 is static. Reading the table:
    flat in k            -> nothing to learn about the rate; make it a knob (κ)
    peak at k ~ 0-2      -> the old "1 patch per step" collapse was near-optimal
    interior optimum     -> a rate worth learning, if the policy can explore it

    python bin/sweep_write_rate.py --cache_dir cache/kvasir_polyppvt_staged_abrupt_4 \\
        --capacity 2000 --warmup 100 --auroc
"""
import csv
import dataclasses
import os
import sys

import click
import numpy as np
import torch  # noqa: F401 — load before faiss (macOS libomp)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import time

import patchcore.metrics
from patchcore.streaming import policies as P
from patchcore.streaming.bank import device_banner, knn_device
from patchcore.streaming.cache import EmbeddingCacheReader
from patchcore.streaming.env import MemoryMaintenanceEnv
from patchcore.streaming.reward import RewardConfig
from patchcore.streaming.teacher import load_teacher


def image_auroc(bank, reader, k, chunk_images=64):
    """Image AUROC only — the same scores as evaluate_bank_on_stage (mean of
    the k smallest squared L2 distances per patch, max over patches) without
    the pixel maps / pixel AUROC / PRO it also computes. Streams the test
    cache in image chunks so nothing large is held in memory."""
    import torch

    dev = knn_device()
    base = torch.from_numpy(bank.vectors()).to(dev)
    kk = min(k, len(base))
    emb = reader.embeddings
    n, p, d = emb.shape
    scores = np.empty(n, dtype=np.float64)
    for s in range(0, n, chunk_images):
        q = torch.from_numpy(np.ascontiguousarray(emb[s:s + chunk_images], dtype=np.float32))
        q = q.to(dev).reshape(-1, d)
        pd = (torch.cdist(q, base) ** 2).topk(kk, dim=1, largest=False).values.mean(1)
        scores[s:s + chunk_images] = pd.reshape(-1, p).max(1).values.double().cpu().numpy()
    return float(patchcore.metrics.compute_imagewise_retrieval_metrics(
        scores, np.asarray(reader.labels, dtype=int))["auroc"])

RULES = {
    "random": lambda k, seed: P.RandomPolicy(k=k, seed=seed),
    "novel": lambda k, seed: P.FIFOPolicy(k=k),
}


@click.command()
@click.option("--cache_dir", required=True)
@click.option("--capacity", type=int, required=True)
@click.option("--warmup", type=int, default=100)
@click.option("--n_nn", type=int, default=5)
@click.option("--teacher", default=None, help="default: <cache_dir>/teacher_k<n_nn>.npz")
@click.option("--heldout_frac", type=float, default=0.05)
@click.option("--rates", default="0,1,2,4,8,16,32,47,100")
@click.option("--rules", default="random,novel")
@click.option("--seeds", default="0,1", help="held-out-set seeds (every rate shares each)")
@click.option("--auroc", is_flag=True, help="also score labelled per-stage AUROC + forgetting")
@click.option("--out", default=None, help="CSV path (default: <cache_dir>/write_rate_sweep.csv)")
def main(cache_dir, capacity, warmup, n_nn, teacher, heldout_frac, rates, rules, seeds,
         auroc, out):
    stream = EmbeddingCacheReader(os.path.join(cache_dir, "stream"))
    teach = load_teacher(teacher or os.path.join(cache_dir, f"teacher_k{n_nn}.npz"))
    test_dir = os.path.join(cache_dir, "test")
    tests = [EmbeddingCacheReader(os.path.join(test_dir, s))
             for s in sorted(d for d in os.listdir(test_dir) if d.startswith("stage_"))]
    print(device_banner())
    if knn_device().type == "cpu":
        print("WARNING: no GPU — every k-NN runs on CPU and the sweep will be very slow. "
              "Run it inside the patchcore conda env on a GPU node (sbatch).")
    rates = [int(r) for r in rates.split(",")]
    seeds = [int(s) for s in seeds.split(",")]

    rows = []
    for rule in rules.split(","):
        for k in rates:
            if k == 0 and rule != rules.split(",")[0]:
                continue  # static is the same under every rule
            for seed in seeds:
                env = MemoryMaintenanceEnv(
                    stream, capacity=capacity, warmup_images=warmup, seed=seed,
                    n_nearest_neighbours=n_nn, reward_cfg=dataclasses.replace(RewardConfig()),
                    reward_mode="iota", teacher=teach, heldout_frac=heldout_frac,
                    heldout_seed=seed, compute_obs=False)
                pol = P.StaticPolicy() if k == 0 else RULES[rule](k, seed)
                stage_auc = {}
                t_start = time.perf_counter()

                def eval_fn(env, stage):
                    if auroc:
                        stage_auc[stage] = image_auroc(env.bank, tests[stage], n_nn)
                    return {}

                summ = P.run_policy(env, pol, per_stage_eval=eval_fn)
                iotas = np.array([i["iota"] for i in summ["infos"]], dtype=float)
                row = {"rule": "static" if k == 0 else rule, "k": k, "seed": seed,
                       "iota_return": float(np.mean(-np.nan_to_num(iotas, nan=0.0))),
                       "final_iota": float(iotas[np.isfinite(iotas)][-1]),
                       "writes": summ["total_admit"]}
                if auroc:
                    row.update({f"auroc_s{s}": v for s, v in stage_auc.items()})
                    row["forget"] = image_auroc(env.bank, tests[0], n_nn)
                rows.append(row)
                print(f"{row['rule']:7s} k={k:3d} seed={seed} ι return {row['iota_return']:+.4f} "
                      f"writes {row['writes']:7d}"
                      + ("  AUROC " + " ".join(f"{row.get(f'auroc_s{s}', float('nan')):.3f}"
                                               for s in range(len(tests)))
                         + f"  forget {row['forget']:.3f}" if auroc else "")
                      + f"  [{time.perf_counter() - t_start:.0f}s]", flush=True)

    out = out or os.path.join(cache_dir, "write_rate_sweep.csv")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("\n=== mean over seeds (ι return: higher is better) ===")
    keys = list(dict.fromkeys((r["rule"], r["k"]) for r in rows))
    for rule, k in keys:
        g = [r for r in rows if r["rule"] == rule and r["k"] == k]
        v = np.array([r["iota_return"] for r in g])
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
        line = f"{rule:7s} k={k:3d}  ι {v.mean():+.4f} ± {se:.4f}  writes {np.mean([r['writes'] for r in g]):9.0f}"
        if auroc:
            for s in range(len(tests)):
                line += f"  s{s} {np.mean([r.get(f'auroc_s{s}', np.nan) for r in g]):.3f}"
            line += f"  forget {np.mean([r['forget'] for r in g]):.3f}"
        print(line)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
