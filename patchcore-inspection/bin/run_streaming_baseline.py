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
import time

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


# results.csv schema: a superset of the original six columns, so DictReader
# consumers (harvester, run_offline_coreset's rewrite) keep working.
FIELDS = ["policy", "seed", "train_seed", "stage", "image_auroc", "pixel_auroc",
          "pro", "writes", "evict", "iota"]


def _parse_ppo_paths(spec):
    """``"PATH"`` -> [("", PATH)]; ``"TS:PATH,TS:PATH"`` -> one entry per
    training seed. A prefix only counts as a train seed when it is an integer,
    so a bare path containing ':' still parses as a single checkpoint."""
    out = []
    for item in (x.strip() for x in spec.split(",")):
        if not item:
            continue
        head, sep, tail = item.partition(":")
        if sep and head.strip().isdigit():
            out.append((head.strip(), tail))
        else:
            out.append(("", item))
    return out


class _LastInfo:
    """Policy wrapper remembering the latest step's info, so the stage-boundary
    eval can log ι (run_policy's eval hook only receives the env)."""

    def __init__(self, policy):
        self.policy, self.name, self.info = policy, policy.name, {}

    def step(self, env):
        out = self.policy.step(env)
        self.info = out[3]
        return out


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
@click.option("--calib_split", type=float, default=0.0,
              help="Report on the test images NOT in the calibration subset "
                   "(same value as fit_reward_weights.py --calib_split). "
                   "0 = legacy: report on the full test set.")
@click.option("--stage0_every_boundary", is_flag=True, default=False,
              help="Also re-score the stage-0 test set at every intermediate "
                   "stage boundary (rows 'stage0_at_<s>': 50%/75% of the stream "
                   "for 4 stages; 100% is the final_forgetting row).")
@click.option("--reward_mode", type=click.Choice(["proxy", "iota"]), default="proxy",
              help="Env reward for baselines (PPO uses its checkpoint's). iota "
                   "logs the normal-score inflation vs the --teacher per stage.")
@click.option("--teacher", default=None,
              help="teacher_k{K}.npz (bin/build_teacher.py); loaded once, "
                   "shared by every env. Required for --reward_mode iota.")
@click.option("--heldout_frac", type=float, default=0.0,
              help="Whole-image held-out fraction withheld from admission for "
                   "EVERY policy (same set per eval seed). Needed by ι and "
                   "myopic_iota.")
@click.option("--admit_cap_abs", type=int, default=None,
              help="Write constraint κ: max admitted patches per image, "
                   "enforced by the env for every policy.")
@click.option("--out", default="streaming_results")
def main(cache_dir, synthetic, capacity, warmup, n_nn, policy_names, ppo_path,
         action_mode, seeds, reward_json, stage0_only, calib_split,
         stage0_every_boundary, reward_mode, teacher, heldout_frac,
         admit_cap_abs, out):
    import dataclasses

    from patchcore.streaming.bank import device_banner
    from patchcore.streaming.reward import RewardConfig, load_reward_weights

    print(device_banner())
    stream, tests, patch_shape, imagesize = _load_readers(cache_dir, synthetic)
    if calib_split:
        from patchcore.streaming.cache import split_test_readers

        tests = split_test_readers(tests, calib_split, "eval")
        print(f"[calib] reporting on {tests[0].n_images} held-out test images "
              f"(calibration subset {calib_split:.0%} excluded)")
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

    if "myopic_iota" in names and not heldout_frac:
        raise click.UsageError("myopic_iota looks ahead on the held-out set: "
                               "pass --heldout_frac > 0")
    if reward_mode == "iota" and not teacher:
        raise click.UsageError("--reward_mode iota requires --teacher")
    teacher_tab = None
    if teacher:
        from patchcore.streaming.teacher import load_teacher

        teacher_tab = load_teacher(teacher)  # once: never per policy x seed
        print(f"[teacher] {teacher}: k={teacher_tab['k']} "
              f"eps={float(teacher_tab['eps']):.4g}")

    def make_env(seed, obs_norm=None, mode=None, meta=None):
        meta = meta or {}
        r_mode = meta.get("reward_mode") or reward_mode
        cap = meta["admit_cap_abs"] if "admit_cap_abs" in meta else admit_cap_abs
        h_frac = heldout_frac
        if r_mode == "iota" and not h_frac:
            h_frac = meta.get("heldout_frac") or 0.0
            print(f"[ppo] checkpoint trained with iota; using its heldout_frac="
                  f"{h_frac} (benchmark passed none)")
        if r_mode == "iota" and teacher_tab is None:
            raise click.UsageError("checkpoint was trained with reward_mode=iota: "
                                   "pass --teacher")
        # New env kwargs only when they deviate from the defaults, so a run
        # without the new flags constructs exactly the legacy env.
        extra = {}
        if r_mode != "proxy":
            extra["reward_mode"] = r_mode
        if teacher_tab is not None:
            extra["teacher"] = teacher_tab
        if h_frac:
            # heldout_seed = eval seed: every policy sees the same H per seed
            extra.update(heldout_frac=h_frac, heldout_seed=seed)
        if cap is not None:
            extra["admit_cap_abs"] = cap
        return MemoryMaintenanceEnv(
            stream, capacity=capacity, warmup_images=warmup, seed=seed,
            action_cfg=ActionConfig(mode=mode or action_mode),
            n_nearest_neighbours=n_nn, obs_norm=obs_norm,
            # fresh copy per env: reset() writes warmup scales into the config
            reward_cfg=dataclasses.replace(reward_cfg),
            **extra,
        )

    s0_mid = []  # (stage, metrics): stage-0 test re-scored mid-stream
    tracker = None  # _LastInfo of the running policy (ι at each boundary)

    # wall-clock split per run: labeled scoring vs the policy's own episode,
    # printed with each run so slow benchmarks show where the time goes
    timing = {"eval": 0.0, "n_eval": 0}

    def timed_eval(*args):
        t0 = time.perf_counter()
        m = evaluate_bank_on_stage(*args)
        timing["eval"] += time.perf_counter() - t0
        timing["n_eval"] += 1
        return m

    def eval_fn(env, stage):
        iota = float(tracker.info.get("iota", float("nan")))
        if stage0_only and stage != 0:
            return {"iota": iota}
        # the last stage's boundary IS the final bank = the forgetting row
        if stage0_every_boundary and 0 < stage < len(tests) - 1:
            s0_mid.append((stage, {"iota": iota, **timed_eval(
                env.bank, tests[0], n_nn, patch_shape, imagesize)}))
        return {"iota": iota, **timed_eval(
            env.bank, tests[stage], n_nn, patch_shape, imagesize
        )}

    def metric_row(name, seed, train_seed, stage, m, summ):
        return {
            "policy": name, "seed": seed, "train_seed": train_seed,
            "stage": stage,
            "image_auroc": m.get("image_auroc", float("nan")),
            "pixel_auroc": m.get("pixel_auroc", float("nan")),
            "pro": m.get("pro", float("nan")),
            # run totals, repeated on every row of the run
            "writes": summ["total_admit"], "evict": summ["total_evict"],
            "iota": m.get("iota", float("nan")),
        }

    rows = []
    for name in names:
        # PPO: one run set per checkpoint (train seed); "" = non-PPO or a
        # single unlabelled checkpoint
        runs = _parse_ppo_paths(ppo_path) if name == "ppo" else [("", None)]
        for train_seed, ckpt in runs:
            if name == "ppo" and not os.path.exists(ckpt):
                print(f"[skip] ppo policy not found at {ckpt}")
                continue
            for seed in seeds:
                # "cpu" here is only the PPO actor-critic's device — a 53-dim MLP
                # queried one obs at a time, where transfer latency exceeds compute.
                # All k-NN / coreset / eval work auto-selects the GPU.
                policy, ppo_norm, ppo_meta = _build_policy(name, ckpt, "cpu")
                # PPO must run under the normalizer + action mode (and reward
                # mode / κ) it trained with; the norm is frozen so sharing
                # across seeds is safe.
                env = make_env(
                    seed, obs_norm=ppo_norm,
                    mode=(ppo_meta or {}).get("action_mode"), meta=ppo_meta,
                )
                s0_mid.clear()
                timing.update(eval=0.0, n_eval=0)
                t_run = time.perf_counter()
                tracker = _LastInfo(policy)
                summ = P.run_policy(env, tracker, per_stage_eval=eval_fn)
                # forgetting: re-evaluate stage-0 test with the final bank (skipped
                # entirely under stage0_only — nothing new to "forget" without
                # drifted stages having run the bank through them)
                forget_m, forget = None, float("nan")
                if not stage0_only:
                    forget_m = timed_eval(
                        env.bank, tests[0], n_nn, patch_shape, imagesize
                    )
                    forget_m["iota"] = float(tracker.info.get("iota", float("nan")))
                    forget = forget_m["image_auroc"]
                for ev in summ["evals"]:
                    rows.append(metric_row(name, seed, train_seed, ev["stage"], ev, summ))
                for st, m in s0_mid:
                    rows.append(metric_row(name, seed, train_seed,
                                           f"stage0_at_{st}", m, summ))
                if not stage0_only:
                    rows.append(metric_row(name, seed, train_seed,
                                           "final_forgetting", forget_m, summ))
                forget_str = f"{forget:.4f}" if not stage0_only else "skipped(stage0_only)"
                ts = f" train_seed={train_seed}" if train_seed != "" else ""
                print(f"{name:26s} seed={seed}{ts} mean_reward={summ['mean_reward']:.4f} "
                      f"admit={summ['total_admit']} evict={summ['total_evict']} "
                      f"stage0_forget={forget_str} "
                      f"[time: episode {time.perf_counter() - t_run - timing['eval']:.0f}s, "
                      f"scoring {timing['eval']:.0f}s over {timing['n_eval']} evals]")

    csv_path = os.path.join(out, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(rows, f, indent=2, default=str)

    _print_summary(rows, names)
    print(f"\nwrote {csv_path}")


def _print_summary(rows, names):
    for key, title in (("image_auroc", "image AUROC"),
                       ("pixel_auroc", "pixel AUROC"),
                       ("pro", "PRO"),
                       ("iota", "iota (lower is better)")):
        if not any(np.isfinite(r.get(key, float("nan"))) for r in rows):
            continue  # metric unavailable (e.g. no masks cached)
        _print_metric_table(rows, names, key, title)


def _print_metric_table(rows, names, key, title):
    print(f"\n=== mean {title} per policy per stage (over seeds) ===")
    stages = sorted({r["stage"] for r in rows if str(r["stage"]).isdigit()},
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
