"""Gate 2 (proxy validation): does the label-free proxy track detection quality?

Generates many diverse bank states and correlates the proxy reward
(-(C + beta*R)) against labeled image AUROC. Passes iff the combined Spearman
correlation magnitude clears ``--min_rho`` with the correct sign.
"""
import json
import os
import sys

import click

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from patchcore.streaming.experiments import run_proxy_correlation


def _load_cache_readers(cache_dir):
    from patchcore.streaming.cache import EmbeddingCacheReader

    stream = EmbeddingCacheReader(os.path.join(cache_dir, "stream"))
    test_dir = os.path.join(cache_dir, "test")
    stages = sorted(d for d in os.listdir(test_dir) if d.startswith("stage_"))
    tests = [EmbeddingCacheReader(os.path.join(test_dir, s)) for s in stages]
    patch_shape = tuple(stream.manifest.get("patch_shape", (0, 0)))
    imagesize = stream.manifest.get("imagesize")
    return stream, tests, patch_shape, imagesize


@click.command()
@click.option("--cache_dir", default=None)
@click.option("--synthetic", is_flag=True)
@click.option("--capacity", type=int, default=2000)
@click.option("--n_nn", type=int, default=1)
@click.option("--min_rho", type=float, default=0.7)
@click.option("--out", default=None)
def main(cache_dir, synthetic, capacity, n_nn, min_rho, out):
    if synthetic:
        from patchcore.streaming.synthetic import (
            SyntheticConfig, make_all_test_stages, make_synthetic_stream,
        )

        cfg = SyntheticConfig()
        stream = make_synthetic_stream(cfg)
        tests = make_all_test_stages(cfg)
        patch_shape, imagesize = None, None
    elif cache_dir:
        stream, tests, patch_shape, imagesize = _load_cache_readers(cache_dir)
    else:
        raise click.UsageError("pass --cache_dir or --synthetic")

    res = run_proxy_correlation(
        stream, tests, capacity, n_nearest_neighbours=n_nn,
        patch_shape=patch_shape or None, imagesize=imagesize,
    )
    passed = abs(res["rho_combined"]) >= min_rho and res["rho_combined"] > 0

    print(f"pairs evaluated       : {res['n_pairs']}")
    print(f"rho(-coverage, AUROC) : {res['rho_coverage']:.4f}")
    print(f"rho(-redundancy,AUROC): {res['rho_redundancy']:.4f}")
    print(f"fitted beta           : {res['best_beta']:.3f}")
    print(f"rho(combined, AUROC)  : {res['rho_combined']:.4f}")
    print(f"per-stage rho(coverage): {res['per_stage_rho_coverage']}")
    print(f"\nGATE 2 {'PASS' if passed else 'FAIL'} (threshold {min_rho})")

    if out:
        with open(out, "w") as f:
            json.dump({**res, "passed": passed}, f, indent=2)
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
