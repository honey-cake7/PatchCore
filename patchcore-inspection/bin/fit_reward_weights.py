"""Fit proxy-reward weights so mean episode reward ranks policies like AUROC.

Runs each baseline policy over the stream once, recording per-step reward
components and labeled per-stage AUROC, then grid-searches
(beta, gamma, churn_coef, churn_budget) offline for the weighting whose mean
episode reward best Spearman-ranks the policies by
``mean(drifted-stage AUROC) + forget_weight * forgetting AUROC``.
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

from patchcore.streaming.experiments import fit_reward_weights, record_policy_traces


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
@click.option("--min_rho", type=float, default=0.7,
              help="Exit non-zero if the best candidate's Spearman rho is below this")
@click.option("--out", default="reward_weights.json")
def main(cache_dir, synthetic, capacity, warmup, n_nn, policy_names, seeds,
         forget_weight, min_rho, out):
    from patchcore.streaming.bank import device_banner

    if not synthetic and not cache_dir:
        raise click.UsageError("pass --cache_dir or --synthetic")
    print(device_banner())
    stream, tests, patch_shape, imagesize = _load_readers(cache_dir, synthetic)

    traces = record_policy_traces(
        stream, tests, capacity, warmup=warmup, n_nearest_neighbours=n_nn,
        patch_shape=patch_shape, imagesize=imagesize,
        policy_names=[n for n in policy_names.split(",") if n],
        seeds=[int(s) for s in seeds.split(",")],
    )
    result = fit_reward_weights(traces, forget_weight=forget_weight)

    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {out}")
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
