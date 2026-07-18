"""Train the self-contained PPO maintenance policy.

Trains on drift trajectories/seeds and evaluates the learned policy's proxy
return against the hand-designed baselines. With ``--synthetic`` it runs fully
data-free on the drifting-Gaussian fixture (the local smoke test); with
``--cache_dir`` it trains on a real cached embedding stream.
"""
import json
import os
import sys

import click
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from patchcore.streaming.env import ActionConfig, MemoryMaintenanceEnv


def _make_synthetic_env_fns(n_env, capacity, warmup, action_mode, base_seed):
    from patchcore.streaming.synthetic import SyntheticConfig, make_synthetic_stream

    fns = []
    for i in range(n_env):
        seed = base_seed + i
        cfg = SyntheticConfig(seed=seed)
        stream = make_synthetic_stream(cfg)

        def fn(stream=stream, seed=seed):
            return MemoryMaintenanceEnv(
                stream, capacity=capacity, warmup_images=warmup, seed=seed,
                action_cfg=ActionConfig(mode=action_mode),
            )

        fns.append(fn)
    return fns


def _make_cache_env_fns(n_env, cache_dir, capacity, warmup, action_mode, base_seed):
    from patchcore.streaming.cache import EmbeddingCacheReader

    reader = EmbeddingCacheReader(os.path.join(cache_dir, "stream"))
    fns = []
    for i in range(n_env):
        def fn(seed=base_seed + i):
            return MemoryMaintenanceEnv(
                reader, capacity=capacity, warmup_images=warmup, seed=seed,
                action_cfg=ActionConfig(mode=action_mode),
            )

        fns.append(fn)
    return fns


@click.command()
@click.option("--cache_dir", default=None)
@click.option("--synthetic", is_flag=True)
@click.option("--n_env", type=int, default=8)
@click.option("--capacity", type=int, default=1500)
@click.option("--warmup", type=int, default=100)
@click.option("--action_mode", default="continuous6",
              type=click.Choice(["continuous6", "bar_only", "discrete4"]))
@click.option("--total_env_steps", type=int, default=200_000)
@click.option("--seed", type=int, default=0)
@click.option("--out", default="ppo_policy.pt")
@click.option("--device", default="cpu", help="Device for the actor-critic net")
@click.option("--eval_baselines", is_flag=True, help="Compare learned proxy return vs baselines")
def main(cache_dir, synthetic, n_env, capacity, warmup, action_mode,
         total_env_steps, seed, out, device, eval_baselines):
    from patchcore.streaming.bank import device_banner
    from patchcore.streaming.ppo import PPOConfig, PPOTrainer

    print(device_banner())
    if synthetic:
        env_fns = _make_synthetic_env_fns(n_env, capacity, warmup, action_mode, seed)
    elif cache_dir:
        env_fns = _make_cache_env_fns(n_env, cache_dir, capacity, warmup, action_mode, seed)
    else:
        raise click.UsageError("pass --cache_dir or --synthetic")

    cfg = PPOConfig(total_env_steps=total_env_steps, device=device)
    trainer = PPOTrainer(env_fns, cfg)
    history = trainer.train()
    trainer.save(out)
    print(f"saved policy -> {out}")

    if eval_baselines:
        _compare_to_baselines(trainer, env_fns[0])


def _compare_to_baselines(trainer, env_fn):
    """Report mean proxy return of PPO vs each baseline on a held-out env."""
    from patchcore.streaming import policies as P
    from patchcore.streaming.policies import PPOPolicy, run_policy

    print("\n=== proxy return (higher is better) ===")
    ppo_ret = np.mean(run_policy(env_fn(), PPOPolicy(trainer.ac))["rewards"])
    print(f"{'ppo':26s} {ppo_ret:.4f}")
    for name, cls in P.BASELINES.items():
        pol = cls(k=8) if name in ("fifo", "random") else cls()
        ret = np.mean(run_policy(env_fn(), pol)["rewards"])
        print(f"{name:26s} {ret:.4f}")


if __name__ == "__main__":
    main()
