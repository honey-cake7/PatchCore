"""Benchmark maintenance policies over a drifting stream (baselines + PPO).

Runs each policy over the cached (or synthetic) stream, freezes the bank at each
stage boundary, and evaluates per-stage image AUROC / pixel AUROC / PRO plus
forgetting (stage-0 test after the full stream) and maintenance cost (bank
mutations). Writes a CSV/JSON summary.
"""
import csv
import json
import os
import sys

import click
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from patchcore.streaming import policies as P
from patchcore.streaming.env import ActionConfig, MemoryMaintenanceEnv
from patchcore.streaming.evaluate import evaluate_bank_on_stage


def _load_readers(cache_dir, synthetic):
    if synthetic:
        from patchcore.streaming.synthetic import (
            SyntheticConfig, make_all_test_stages, make_synthetic_stream,
        )

        cfg = SyntheticConfig()
        return make_synthetic_stream(cfg), make_all_test_stages(cfg), None, None
    from patchcore.streaming.cache import EmbeddingCacheReader

    stream = EmbeddingCacheReader(os.path.join(cache_dir, "stream"))
    test_dir = os.path.join(cache_dir, "test")
    stages = sorted(d for d in os.listdir(test_dir) if d.startswith("stage_"))
    tests = [EmbeddingCacheReader(os.path.join(test_dir, s)) for s in stages]
    patch_shape = tuple(stream.manifest.get("patch_shape", (0, 0))) or None
    imagesize = stream.manifest.get("imagesize")
    return stream, tests, patch_shape, imagesize


def _build_policy(name, ppo_path, device):
    """Returns (policy, obs_norm, meta). obs_norm/meta are None except for PPO
    v2 checkpoints, which carry the normalizer + action mode they trained under."""
    if name == "ppo":
        from patchcore.streaming.policies import PPOPolicy
        from patchcore.streaming.ppo import load_checkpoint

        ac, norm, meta = load_checkpoint(ppo_path, device=device)
        return PPOPolicy(ac, device=device), norm, meta
    cls = P.BASELINES[name]
    return (cls(k=8) if name in ("fifo", "random") else cls()), None, None


@click.command()
@click.option("--cache_dir", default=None)
@click.option("--synthetic", is_flag=True)
@click.option("--capacity", type=int, default=2000)
@click.option("--warmup", type=int, default=100)
@click.option("--n_nn", type=int, default=1)
@click.option("--policies", "policy_names", default="static,fifo,reservoir,streaming_greedy_coreset,periodic_coreset,ppo")
@click.option("--ppo_path", default="ppo_policy.pt")
@click.option("--action_mode", default="continuous6")
@click.option("--seeds", default="0,1,2")
@click.option("--reward_json", default=None,
              help="Fitted reward weights (bin/fit_reward_weights.py output)")
@click.option(
    "--stage0_only", is_flag=True, default=False,
    help="Evaluate stage 0 only and skip the final stage-0 forgetting "
         "re-eval — the paper reports stage-0 AUROC only. Episodes still run "
         "the full stream; only the per-stage labeled scoring is pruned.",
)
@click.option("--out", default="streaming_results")
def main(cache_dir, synthetic, capacity, warmup, n_nn, policy_names, ppo_path,
         action_mode, seeds, reward_json, stage0_only, out):
    import dataclasses

    from patchcore.streaming.bank import device_banner
    from patchcore.streaming.reward import RewardConfig, load_reward_weights

    print(device_banner())
    stream, tests, patch_shape, imagesize = _load_readers(cache_dir, synthetic)
    seeds = [int(s) for s in seeds.split(",")]
    names = [n for n in policy_names.split(",") if n]
    os.makedirs(out, exist_ok=True)
    reward_cfg = load_reward_weights(reward_json) if reward_json else RewardConfig()
    if reward_json:
        print(f"[reward] weights from {reward_json}: alpha={reward_cfg.alpha} "
              f"beta={reward_cfg.beta} gamma={reward_cfg.gamma} "
              f"churn_coef={reward_cfg.churn_coef} churn_budget={reward_cfg.churn_budget} "
              f"c90_coef={reward_cfg.c90_coef} probe_coef={reward_cfg.probe_coef} "
              f"q_coef={reward_cfg.q_coef}")

    def make_env(seed, obs_norm=None, mode=None):
        return MemoryMaintenanceEnv(
            stream, capacity=capacity, warmup_images=warmup, seed=seed,
            action_cfg=ActionConfig(mode=mode or action_mode),
            n_nearest_neighbours=n_nn, obs_norm=obs_norm,
            # fresh copy per env: reset() writes warmup scales into the config
            reward_cfg=dataclasses.replace(reward_cfg),
        )

    def eval_fn(env, stage):
        if stage0_only and stage != 0:
            return {}
        return evaluate_bank_on_stage(
            env.bank, tests[stage], n_nn, patch_shape, imagesize
        )

    rows = []
    for name in names:
        if name == "ppo" and not os.path.exists(ppo_path):
            print(f"[skip] ppo policy not found at {ppo_path}")
            continue
        for seed in seeds:
            # "cpu" here is only the PPO actor-critic's device — a 53-dim MLP
            # queried one obs at a time, where transfer latency exceeds compute.
            # All k-NN / coreset / eval work auto-selects the GPU.
            policy, ppo_norm, ppo_meta = _build_policy(name, ppo_path, "cpu")
            # PPO must run under the normalizer + action mode it trained with
            # (v2 checkpoints); the norm is frozen so sharing across seeds is safe.
            env = make_env(
                seed, obs_norm=ppo_norm,
                mode=(ppo_meta or {}).get("action_mode"),
            )
            summ = P.run_policy(env, policy, per_stage_eval=eval_fn)
            # forgetting: re-evaluate stage-0 test with the final bank (skipped
            # entirely under stage0_only — nothing new to "forget" without
            # drifted stages having run the bank through them)
            forget_m, forget = None, float("nan")
            if not stage0_only:
                forget_m = evaluate_bank_on_stage(
                    env.bank, tests[0], n_nn, patch_shape, imagesize
                )
                forget = forget_m["image_auroc"]
            for ev in summ["evals"]:
                rows.append({
                    "policy": name, "seed": seed, "stage": ev["stage"],
                    "image_auroc": ev.get("image_auroc", float("nan")),
                    "pixel_auroc": ev.get("pixel_auroc", float("nan")),
                    "pro": ev.get("pro", float("nan")),
                })
            if not stage0_only:
                rows.append({
                    "policy": name, "seed": seed, "stage": "final_forgetting",
                    "image_auroc": forget,
                    "pixel_auroc": forget_m.get("pixel_auroc", float("nan")),
                    "pro": forget_m.get("pro", float("nan")),
                })
            forget_str = f"{forget:.4f}" if not stage0_only else "skipped(stage0_only)"
            print(f"{name:26s} seed={seed} mean_reward={summ['mean_reward']:.4f} "
                  f"admit={summ['total_admit']} evict={summ['total_evict']} "
                  f"stage0_forget={forget_str}")

    csv_path = os.path.join(out, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["policy", "seed", "stage", "image_auroc",
                                          "pixel_auroc", "pro"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(rows, f, indent=2, default=str)

    _print_summary(rows, names)
    print(f"\nwrote {csv_path}")


def _print_summary(rows, names):
    for key, title in (("image_auroc", "image AUROC"),
                       ("pixel_auroc", "pixel AUROC"),
                       ("pro", "PRO")):
        if not any(np.isfinite(r.get(key, float("nan"))) for r in rows):
            continue  # metric unavailable (e.g. no masks cached)
        _print_metric_table(rows, names, key, title)


def _print_metric_table(rows, names, key, title):
    print(f"\n=== mean {title} per policy per stage (over seeds) ===")
    stages = sorted({r["stage"] for r in rows if r["stage"] != "final_forgetting"},
                    key=lambda x: int(x))
    header = "policy".ljust(26) + "".join(f"stage{ s}".rjust(10) for s in stages) + "forget".rjust(10)
    print(header)
    for name in names:
        cells = []
        for s in stages:
            vals = [r[key] for r in rows
                    if r["policy"] == name and r["stage"] == s
                    and np.isfinite(r.get(key, float("nan")))]
            cells.append(f"{np.mean(vals):.3f}".rjust(10) if vals else "n/a".rjust(10))
        fvals = [r[key] for r in rows
                 if r["policy"] == name and r["stage"] == "final_forgetting"
                 and np.isfinite(r.get(key, float("nan")))]
        forget = f"{np.mean(fvals):.3f}".rjust(10) if fvals else "n/a".rjust(10)
        print(name.ljust(26) + "".join(cells) + forget)


if __name__ == "__main__":
    main()
