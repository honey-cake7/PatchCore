"""Build the full-memory teacher table for the ι reward (once per cache and k).

Writes ``<cache_dir>/teacher_k{K}.npz`` (see patchcore.streaming.teacher).
Every policy x seed run then loads the table instead of recomputing it.

Near-duplicate clusters need the original image files: pass --data_path and
--classname (drift/seed default to the cache manifest). The cached stream
order is reconstructed exactly like bin/cache_embeddings.py; if the count does
not match the cache, clustering falls back to singletons with a warning.

Example::

    python bin/build_teacher.py --cache_dir cache/kvasir/wrn50/staged_abrupt_4 \
        --n_nn 5 --data_path /data/kvasir_patchcore --classname kvasir_patchcore
"""
import os
import sys
import time

import click
import numpy as np
import torch  # noqa: F401  (import torch before anything that may pull faiss)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from patchcore.streaming.bank import device_banner
from patchcore.streaming.cache import EmbeddingCacheReader
from patchcore.streaming.teacher import build_teacher, dhash_clusters, save_teacher


def _stream_clusters(n_images, manifest, data_path, classname, drift, seed):
    from patchcore.streaming import drift as drift_mod
    from patchcore.streaming.stream import StreamBuilder

    builder = StreamBuilder(data_path, classname)
    if manifest.get("drift_mode", "synthetic") == "real":
        entries = builder.real_drift_stream()
    else:
        entries = builder.synthetic_drift_stream(drift_mod.SCHEDULES[drift], seed=seed)
    if len(entries) != n_images:
        print(f"[teacher] WARNING: reconstructed stream has {len(entries)} images, "
              f"cache has {n_images} — falling back to singleton clusters")
        return None
    clusters = dhash_clusters([e.image_path for e in entries])
    n_cl = len(np.unique(clusters))
    print(f"[teacher] dHash clusters: {n_cl} clusters over {n_images} images "
          f"({n_images - n_cl} near-duplicates merged)")
    return clusters


@click.command()
@click.option("--cache_dir", required=True)
@click.option("--n_nn", type=int, default=5, help="k of the k-NN scorer (match eval)")
@click.option("--data_path", default=None, help="Original dataset root (for clustering)")
@click.option("--classname", default=None)
@click.option("--drift", "drift_name", default=None, help="Default: cache manifest")
@click.option("--seed", type=int, default=None, help="Default: cache manifest")
@click.option("--block", type=int, default=32, help="Stream images per distance block")
@click.option("--out", default=None, help="Default: <cache_dir>/teacher_k{K}.npz")
def main(cache_dir, n_nn, data_path, classname, drift_name, seed, block, out):
    print(device_banner())
    reader = EmbeddingCacheReader(os.path.join(cache_dir, "stream"))
    manifest = reader.manifest
    clusters = None
    if data_path and classname:
        clusters = _stream_clusters(
            reader.n_images, manifest, data_path, classname,
            drift_name or manifest.get("drift", "staged_abrupt_4"),
            seed if seed is not None else int(manifest.get("seed", 0)))
    else:
        print("[teacher] no --data_path/--classname: singleton clusters")
    t0 = time.perf_counter()
    teacher = build_teacher(reader, n_nn, clusters=clusters, block=block)
    print(f"[teacher] built in {time.perf_counter() - t0:.1f}s "
          f"(N={teacher['n_images']}, P={reader.patches_per_image}, D={reader.dim}); "
          f"eps={teacher['eps']:.4g}")
    out = out or os.path.join(cache_dir, f"teacher_k{n_nn}.npz")
    save_teacher(out, teacher)
    print(f"[teacher] wrote {out}")


if __name__ == "__main__":
    main()
