"""Deployment cost of one streaming bank: per-image latency and peak GPU memory.

Latency per image, measured separately (CUDA-synchronized, after warm-up):
    feature_ms      frozen backbone forward + patch embedding (real test images)
    nn_search_ms    k-NN anomaly scoring of one image's patches against the bank
    maintenance_ms  one env step per policy: novelty k-NN + decision + bank
                    update + the label-free proxy bookkeeping the observation
                    reads (an upper bound on what deployment needs)
Peak GPU memory is torch.cuda.max_memory_allocated with backbone, bank, policy,
recent window and probe set Z all resident. Writes <out>/deployment.csv.
"""
import csv
import dataclasses
import os
import sys
import time

import click
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import PIL.Image


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(fn, n, warm=5):
    for _ in range(min(warm, n)):
        fn()
    _sync()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    _sync()
    return 1e3 * (time.perf_counter() - t) / n


@click.command()
@click.option("--backbone_name", required=True)
@click.option("--layers_to_extract_from", "-le", multiple=True, required=True)
@click.option("--data_path", required=True)
@click.option("--classname", required=True)
@click.option("--resize", type=int, default=256)
@click.option("--imagesize", type=int, default=224)
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--patchsize", type=int, default=3)
@click.option("--cache_dir", required=True)
@click.option("--capacity", type=int, required=True)
@click.option("--warmup", type=int, default=100)
@click.option("--n_nn", type=int, default=5)
@click.option("--ppo_path", default=None)
@click.option("--reward_json", default=None)
@click.option("--policies", default="static,fifo,reservoir,streaming_greedy_coreset,periodic_coreset,kcenter,myopic_iota,ppo")
@click.option("--reward_mode", default="proxy", type=click.Choice(["proxy", "iota"]))
@click.option("--teacher", default=None)
@click.option("--heldout_frac", type=float, default=0.0)
@click.option("--admit_cap_abs", type=int, default=None)
@click.option("--n_images", type=int, default=50)
@click.option("--n_steps", type=int, default=100)
@click.option("--out", required=True)
def main(backbone_name, layers_to_extract_from, data_path, classname, resize,
         imagesize, pretrain_embed_dimension, target_embed_dimension, patchsize,
         cache_dir, capacity, warmup, n_nn, ppo_path, reward_json, policies,
         reward_mode, teacher, heldout_frac, admit_cap_abs, n_images, n_steps, out):
    from cache_embeddings import _base_transforms, _embed_image, _to_tensor, build_patchcore

    from patchcore.streaming import policies as P
    from patchcore.streaming.cache import EmbeddingCacheReader
    from patchcore.streaming.env import ActionConfig, MemoryMaintenanceEnv
    from patchcore.streaming.evaluate import _torch_patch_scores
    from patchcore.streaming.reward import RewardConfig, load_reward_weights
    from patchcore.streaming.stream import StreamBuilder

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ---- backbone + feature extraction ----
    model = build_patchcore(backbone_name, layers_to_extract_from, dev, imagesize,
                            pretrain_embed_dimension, target_embed_dimension, patchsize)
    base_tf, to_tensor = _base_transforms(resize, imagesize), _to_tensor()
    items = StreamBuilder(data_path, classname)._test_items()[:n_images]
    imgs = [base_tf(PIL.Image.open(it.image_path).convert("RGB")) for it in items]
    it_img = iter(imgs * (1 + (n_images + 5) // max(1, len(imgs))))
    feature_ms = _timed(lambda: _embed_image(model, next(it_img), to_tensor, dev), len(imgs))
    backbone_mib = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20

    # ---- bank (warmup coreset at M) + NN search ----
    stream = EmbeddingCacheReader(os.path.join(cache_dir, "stream"))
    reward_cfg = load_reward_weights(reward_json) if reward_json else RewardConfig()
    # ι mode: the maintenance timing includes the per-step ι tracker, i.e. what
    # training pays; a deployed policy needs neither teacher nor reward, so this
    # is an upper bound. myopic_iota needs the held-out set (heldout_frac > 0).
    extra = {}
    if heldout_frac:
        extra.update(heldout_frac=heldout_frac, heldout_seed=0)
    if admit_cap_abs is not None:
        extra["admit_cap_abs"] = admit_cap_abs
    if reward_mode == "iota":
        from patchcore.streaming.teacher import load_teacher

        extra.update(reward_mode="iota", teacher=load_teacher(teacher))
    env0 = MemoryMaintenanceEnv(stream, capacity=capacity, warmup_images=warmup,
                                n_nearest_neighbours=n_nn,
                                reward_cfg=dataclasses.replace(reward_cfg), **extra)
    env0.reset()
    test = EmbeddingCacheReader(os.path.join(cache_dir, "test", "stage_0"))
    q = [np.ascontiguousarray(test.embeddings[i], dtype=np.float32)
         for i in range(min(n_images, test.n_images))]
    it_q = iter(q * (1 + (n_images + 5) // max(1, len(q))))
    nn_ms = _timed(lambda: _torch_patch_scores(env0.bank, next(it_q), n_nn), len(q))

    # ---- maintenance step per policy ----
    rows = []
    for name in [n for n in policies.split(",") if n]:
        norm = mode = None
        if name == "ppo":
            if not ppo_path or not os.path.exists(ppo_path):
                print(f"[skip] ppo checkpoint not found: {ppo_path}")
                continue
            from patchcore.streaming.ppo import load_checkpoint

            ac, norm, meta = load_checkpoint(ppo_path, device="cpu")
            pol, mode = P.PPOPolicy(ac), (meta or {}).get("action_mode")
        else:
            if name == "myopic_iota" and not heldout_frac:
                print("[skip] myopic_iota needs --heldout_frac > 0")
                continue
            cls = P.BASELINES[name]
            pol = cls(k=8) if name in ("fifo", "random") else cls()
        env = MemoryMaintenanceEnv(stream, capacity=capacity, warmup_images=warmup,
                                   n_nearest_neighbours=n_nn, obs_norm=norm,
                                   action_cfg=ActionConfig(mode=mode or "continuous6"),
                                   reward_cfg=dataclasses.replace(reward_cfg), **extra)
        env.import_init_state(env0.export_init_state())
        env.reset()
        steps = min(n_steps, stream.n_images - warmup - 1)
        # periodic_coreset's cost is concentrated in every 50th step, so time a
        # whole window of steps (not a warm subset) and report the mean
        maint_ms = _timed(lambda: pol.step(env), steps, warm=0)
        rows.append({"policy": name, "maintenance_ms": round(maint_ms, 3)})
        print(f"{name:26s} maintenance {maint_ms:.2f} ms/step")

    peak = torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else float("nan")
    D = stream.dim
    common = {"gpu": gpu, "M": capacity, "D": D, "feature_ms": round(feature_ms, 3),
              "nn_search_ms": round(nn_ms, 3), "backbone_MiB": round(backbone_mib, 1),
              "bank_MiB": round(capacity * D * 4 / 2**20, 2),
              "peak_gpu_MiB": round(peak, 1)}
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "deployment.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(common) + ["policy", "maintenance_ms", "total_ms"])
        w.writeheader()
        for r in rows:
            w.writerow({**common, **r, "total_ms": round(
                feature_ms + nn_ms + r["maintenance_ms"], 3)})
    print(f"feature {feature_ms:.2f} ms | nn search {nn_ms:.2f} ms | "
          f"peak GPU {peak:.0f} MiB -> {path}")


if __name__ == "__main__":
    main()
