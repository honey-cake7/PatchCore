"""Fit proxy-reward weights so mean episode reward ranks policies like AUROC.

Runs each baseline policy over the stream once, recording per-step reward
components and labeled per-stage AUROC, then grid-searches
(beta, gamma, churn_coef, churn_budget) offline for the weighting whose mean
episode reward best Spearman-ranks the policies by
``drifted_weight * mean(drifted-stage AUROC) + forget_weight * forgetting
AUROC + stage0_weight * stage-0 AUROC``.
Writes ``reward_weights.json``, consumed by train_ppo.py /
run_streaming_baseline.py via ``--reward_json``.
"""
import json
import os
import sys

import click
import torch  # noqa: F401 — must load before faiss (via patchcore.common) so a
              # single OpenMP runtime wins; reversed order segfaults on macOS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from patchcore.streaming.experiments import (
    fit_reward_weights,
    record_policy_traces,
    record_ppo_traces,
)


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


@click.command()
@click.option("--cache_dir", default=None)
@click.option("--synthetic", is_flag=True)
@click.option("--capacity", type=int, default=2000)
@click.option("--warmup", type=int, default=100)
@click.option("--n_nn", type=int, default=1)
@click.option("--policies", "policy_names",
              default="static,fifo,random,reservoir,streaming_greedy_coreset,periodic_coreset")
@click.option("--seeds", default="0,1")
@click.option("--forget_weight", type=float, default=0.5,
              help="Weight of the stage-0 forgetting AUROC in the ranking target")
@click.option("--drifted_weight", type=float, default=1.0,
              help="Weight of the mean drifted-stage (stage>=1) AUROC in the target")
@click.option("--stage0_weight", type=float, default=0.0,
              help="Weight of the stage-0 AUROC in the target. The default "
                   "target excludes stage 0, so weights fitted under it do not "
                   "optimize the stage-0-vs-stock comparison. Stage-0 AUROC is "
                   "already in every trace: re-target with --traces_in in "
                   "seconds, no re-recording. Stage-0 only: "
                   "--stage0_weight 1 --drifted_weight 0 --forget_weight 0")
@click.option("--min_rho", type=float, default=0.7,
              help="Exit non-zero if the best candidate's Spearman rho is below this")
@click.option("--traces_out", default=None,
              help="Pickle recorded traces here (default: <out dir>/reward_traces.pkl)")
@click.option("--traces_in", default=None,
              help="Refit from previously recorded traces; skips the stream passes")
@click.option("--ppo_pt", default=None,
              help="Trained PPO checkpoint: replay it, ADD its traces to the "
                   "fitting set, and persist the merged traces (iterated "
                   "refit — closes proxy directions the heuristic fitting set "
                   "never visited). Needs --cache_dir/--synthetic for readers.")
@click.option("--out", default="reward_weights.json")
def main(cache_dir, synthetic, capacity, warmup, n_nn, policy_names, seeds,
         forget_weight, drifted_weight, stage0_weight, min_rho, traces_out,
         traces_in, ppo_pt, out):
    import pickle

    from patchcore.streaming.bank import device_banner

    seed_list = [int(s) for s in seeds.split(",")]
    default_traces = os.path.join(
        os.path.dirname(os.path.abspath(out)), "reward_traces.pkl")
    stream = tests = patch_shape = imagesize = None
    if not traces_in or ppo_pt:
        if not synthetic and not cache_dir:
            raise click.UsageError(
                "pass --cache_dir or --synthetic (needed to record traces)")
        print(device_banner())
        stream, tests, patch_shape, imagesize = _load_readers(cache_dir, synthetic)

    if traces_in:
        with open(traces_in, "rb") as f:
            traces = pickle.load(f)
        print(f"loaded {len(traces)} traces from {traces_in} (skipping stream passes)")
    else:
        traces = record_policy_traces(
            stream, tests, capacity, warmup=warmup, n_nearest_neighbours=n_nn,
            patch_shape=patch_shape, imagesize=imagesize,
            policy_names=[n for n in policy_names.split(",") if n],
            seeds=seed_list,
        )

    if ppo_pt:
        # Auto-number the refit round from how many ppo trace sets exist, so
        # repeated rounds accumulate (ppo_r1, ppo_r2, ...) instead of colliding.
        rounds = {t["policy"] for t in traces if str(t["policy"]).startswith("ppo_r")}
        name = f"ppo_r{len(rounds) + 1}"
        print(f"recording trained-policy traces from {ppo_pt} as '{name}' ...")
        traces = traces + record_ppo_traces(
            stream, tests, capacity, ppo_pt, warmup=warmup,
            n_nearest_neighbours=n_nn, patch_shape=patch_shape,
            imagesize=imagesize, seeds=seed_list, name=name,
        )

    if not traces_in or ppo_pt:
        traces_out = traces_out or traces_in or default_traces
        with open(traces_out, "wb") as f:
            pickle.dump(traces, f)
        print(f"saved {len(traces)} traces -> {traces_out} "
              "(refit later with --traces_in)")
    result = fit_reward_weights(
        traces, forget_weight=forget_weight,
        drifted_weight=drifted_weight, stage0_weight=stage0_weight,
    )

    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {out}")
    print(f"target: {drifted_weight}*drifted + {forget_weight}*forgetting "
          f"+ {stage0_weight}*stage0")
    print(f"recommended weights : {result['recommended']}")
    print(f"rho_ranking (fitted): {result['rho_ranking']:.3f}")
    print(f"rho_ranking (current weights): {result['rho_ranking_current_weights']:.3f}")

    if result["rho_ranking"] < min_rho:
        print(f"FAIL: fitted rho {result['rho_ranking']:.3f} < min_rho {min_rho} — "
              "proxy cannot rank policies like AUROC; do not retrain on it.")
        sys.exit(2)
    print(f"PASS: fitted rho >= {min_rho}")


if __name__ == "__main__":
    main()
