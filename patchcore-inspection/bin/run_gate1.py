"""Gate 1 (headroom): does drift actually hurt a static memory bank?

Runs on a written embedding cache (``--cache_dir`` from cache_embeddings.py) or,
with ``--synthetic``, on the drifting-Gaussian fixture for a data-free check.
Passes iff the mean oracle-minus-static AUROC gap over drifted stages exceeds
``--min_gap``.
"""
import json
import os
import sys

import click

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from patchcore.streaming.cache import EmbeddingCacheReader
from patchcore.streaming.experiments import headroom_gap, run_headroom


def _load_cache_readers(cache_dir):
    stream = EmbeddingCacheReader(os.path.join(cache_dir, "stream"))
    test_dir = os.path.join(cache_dir, "test")
    stages = sorted(
        d for d in os.listdir(test_dir) if d.startswith("stage_")
    )
    tests = [EmbeddingCacheReader(os.path.join(test_dir, s)) for s in stages]
    patch_shape = tuple(stream.manifest.get("patch_shape", (0, 0)))
    imagesize = stream.manifest.get("imagesize")
    return stream, tests, patch_shape, imagesize


@click.command()
@click.option("--cache_dir", default=None, help="Directory written by cache_embeddings.py")
@click.option("--synthetic", is_flag=True, help="Use the drifting-Gaussian fixture")
@click.option("--capacity", type=int, default=2000)
@click.option("--n_nn", type=int, default=1)
@click.option("--min_gap", type=float, default=0.03)
@click.option("--out", default=None, help="Optional JSON output path")
def main(cache_dir, synthetic, capacity, n_nn, min_gap, out):
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

    results = run_headroom(
        stream, tests, capacity, n_nearest_neighbours=n_nn,
        patch_shape=patch_shape or None, imagesize=imagesize,
    )
    gap = headroom_gap(results)
    passed = gap >= min_gap

    print(f"{'stage':>5} {'static_auroc':>13} {'oracle_auroc':>13} "
          f"{'static_pro':>11} {'oracle_pro':>11}")
    for r in results:
        print(f"{r['stage']:>5} {r['static_auroc']:>13.4f} {r['oracle_auroc']:>13.4f} "
              f"{r['static_pro']:>11.4f} {r['oracle_pro']:>11.4f}")
    print(f"\nMean oracle-static AUROC gap (drifted stages): {gap:.4f}")
    print(f"GATE 1 {'PASS' if passed else 'FAIL'} (threshold {min_gap})")

    if out:
        with open(out, "w") as f:
            json.dump({"results": results, "gap": gap, "passed": passed}, f, indent=2)
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
