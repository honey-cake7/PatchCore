"""Train the self-contained PPO maintenance policy.

Trains on drift trajectories/seeds and evaluates the learned policy's proxy
return against the hand-designed baselines. With ``--synthetic`` it runs fully
data-free on the drifting-Gaussian fixture (the local smoke test); with
``--cache_dir`` it trains on a real cached embedding stream.
"""
import dataclasses
import json
import os
import sys

import click
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from patchcore.streaming.env import ActionConfig, MemoryMaintenanceEnv
from patchcore.streaming.reward import RewardConfig, load_reward_weights


def _make_synthetic_env_fns(n_env, capacity, warmup, action_mode, base_seed, reward_cfg):
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
                # each env gets its own copy: reset() writes warmup scales into it
                reward_cfg=dataclasses.replace(reward_cfg),
            )

        fns.append(fn)
    return fns


def _make_cache_env_fns(n_env, cache_dir, capacity, warmup, action_mode, base_seed, reward_cfg):
    from patchcore.streaming.cache import EmbeddingCacheReader

    reader = EmbeddingCacheReader(os.path.join(cache_dir, "stream"))

    # Every env replays the same stream, so the expensive warmup coreset (and
    # probe/projection/reward scales) is computed once on a prototype and
    # shared; each env keeps its own seed for per-step randomness.
    print("[env] computing warmup coreset once (shared across envs) ...")
    proto = MemoryMaintenanceEnv(
        reader, capacity=capacity, warmup_images=warmup, seed=base_seed,
        action_cfg=ActionConfig(mode=action_mode),
        reward_cfg=dataclasses.replace(reward_cfg),
    )
    proto.reset()
    init_state = proto.export_init_state()

    fns = []
    for i in range(n_env):
        def fn(seed=base_seed + i):
            env = MemoryMaintenanceEnv(
                reader, capacity=capacity, warmup_images=warmup, seed=seed,
                action_cfg=ActionConfig(mode=action_mode),
                reward_cfg=dataclasses.replace(reward_cfg),
            )
            env.import_init_state(init_state)
            return env

        fns.append(fn)
    return fns


def _build_reward_cfg(reward_json, beta, gamma, churn_coef, churn_budget):
    cfg = load_reward_weights(reward_json) if reward_json else RewardConfig()
    if reward_json:
        print(f"[reward] loaded weights from {reward_json}: "
              f"alpha={cfg.alpha} beta={cfg.beta} gamma={cfg.gamma} "
              f"churn_coef={cfg.churn_coef} churn_budget={cfg.churn_budget} "
              f"c90_coef={cfg.c90_coef} probe_coef={cfg.probe_coef} "
              f"q_coef={cfg.q_coef}")
    for name, val in (("beta", beta), ("gamma", gamma),
                      ("churn_coef", churn_coef), ("churn_budget", churn_budget)):
        if val is not None:
            setattr(cfg, name, val)
            print(f"[reward] override {name}={val}")
    return cfg


def _fit_obs_norm(env_fn, seed=0):
    """Fit the observation normalizer over one full random-policy episode.

    The default RunningNorm freezes after 128 stage-0 steps, so its statistics
    never see drifted stages. Prefitting over a whole episode (all stages) and
    freezing gives training and eval the exact same, stationary normalization.
    """
    from patchcore.streaming.policies import RandomPolicy, run_policy

    env = env_fn()
    env.obs_cfg.warmup_steps = 10 ** 9  # never freeze mid-episode
    print("[norm] fitting observation normalizer over one random-policy episode ...")
    run_policy(env, RandomPolicy(k=8, seed=seed))
    norm = env._norm
    norm.freeze()
    print(f"[norm] fitted on {norm.count} steps, frozen")
    return norm


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
@click.option("--reward_json", default=None,
              help="Fitted reward weights (bin/fit_reward_weights.py output)")
@click.option("--beta", type=float, default=None, help="Override reward beta")
@click.option("--gamma", type=float, default=None, help="Override reward gamma")
@click.option("--churn_coef", type=float, default=None, help="Override churn coefficient")
@click.option("--churn_budget", type=float, default=None, help="Override churn budget")
@click.option("--reward_form", default="level", type=click.Choice(["level", "delta"]),
              help="delta = potential-based shaping: reward the step change in "
                   "state cost (action-local signal; same optimal policy)")
@click.option("--norm_mode", default="episode", type=click.Choice(["episode", "running"]),
              help="Obs normalization: prefit+frozen over a full episode, or legacy running warmup")
@click.option("--reward_scale", default="fixed", type=click.Choice(["ewma", "fixed", "none"]))
@click.option("--adv_mode", default="gae", type=click.Choice(["gae", "grpo"]))
@click.option("--clip_mode", default="clip", type=click.Choice(["clip", "gppo"]),
              help="gppo = gradient-preserving clip (clipped samples keep a bounded gradient)")
@click.option("--clip_high", type=float, default=None,
              help="Decoupled upper clip epsilon (clip-higher); default = --clip value")
@click.option("--ent_coef", type=float, default=1e-3)
@click.option("--ent_coef_end", type=float, default=None, help="Linear entropy anneal target")
@click.option("--lr", type=float, default=3e-4)
@click.option("--lr_end", type=float, default=None, help="Linear LR anneal target")
def main(cache_dir, synthetic, n_env, capacity, warmup, action_mode,
         total_env_steps, seed, out, device, eval_baselines, reward_json,
         beta, gamma, churn_coef, churn_budget, reward_form, norm_mode,
         reward_scale, adv_mode, clip_mode, clip_high, ent_coef, ent_coef_end,
         lr, lr_end):
    import torch

    from patchcore.streaming.bank import device_banner
    from patchcore.streaming.ppo import PPOConfig, PPOTrainer

    # Torch sizes its CPU thread pool from the node's core count, not the
    # SLURM cgroup — on a shared node every tiny actor-critic op then forks an
    # oversubscribed thread pool and stalls (observed: ~100ms per 8x53 MLP
    # forward). The ops here are far too small to parallelize: threads=1
    # removes OMP fork/join entirely and is immune to core contention
    # (act still bounced 2s->12s between runs at threads=4 on a busy node).
    n_threads = int(os.environ.get("STREAMING_TORCH_THREADS", "1"))
    torch.set_num_threads(n_threads)
    print(device_banner())
    print(f"[torch] cpu threads capped at {n_threads} "
          f"(STREAMING_TORCH_THREADS to override)")
    reward_cfg = _build_reward_cfg(reward_json, beta, gamma, churn_coef, churn_budget)
    reward_cfg.reward_form = reward_form
    if reward_form == "delta":
        print("[reward] potential-based delta shaping enabled")
    if synthetic:
        env_fns = _make_synthetic_env_fns(
            n_env, capacity, warmup, action_mode, seed, reward_cfg)
    elif cache_dir:
        env_fns = _make_cache_env_fns(
            n_env, cache_dir, capacity, warmup, action_mode, seed, reward_cfg)
    else:
        raise click.UsageError("pass --cache_dir or --synthetic")

    obs_norm = _fit_obs_norm(env_fns[0], seed=seed) if norm_mode == "episode" else None

    cfg = PPOConfig(
        total_env_steps=total_env_steps, device=device,
        reward_scale=reward_scale, adv_mode=adv_mode,
        clip_mode=clip_mode, clip_high=clip_high,
        ent_coef=ent_coef, ent_coef_end=ent_coef_end, lr=lr, lr_end=lr_end,
    )
    if cache_dir:
        # The best-checkpoint score is a rolling mean over one full stream
        # replay; the window must track this dataset's episode length or the
        # selection inherits stream-phase bias (default 9 fits only kvasir).
        # Floor of 3: streams shorter than a rollout (small MVTec classes)
        # would otherwise select on a single noisy iteration.
        from patchcore.streaming.cache import EmbeddingCacheReader

        episode_steps = EmbeddingCacheReader(
            os.path.join(cache_dir, "stream")).n_images - warmup
        cfg.best_window = max(3, round(episode_steps / cfg.rollout_steps))
    print(f"[ppo] adv_mode={adv_mode} clip_mode={clip_mode} clip_high={clip_high} "
          f"best_window={cfg.best_window}")
    # Cache-path envs replay the same stream in lockstep -> fused cross-env
    # k-NN batching in collect(); synthetic envs have per-seed streams.
    trainer = PPOTrainer(env_fns, cfg, obs_norm=obs_norm, lockstep=bool(cache_dir))
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
    ppo_env = env_fn()
    ppo_env._norm = trainer.obs_norm  # eval under the training normalization
    ppo_ret = np.mean(run_policy(ppo_env, PPOPolicy(trainer.ac))["rewards"])
    print(f"{'ppo':26s} {ppo_ret:.4f}")
    for name, cls in P.BASELINES.items():
        pol = cls(k=8) if name in ("fifo", "random") else cls()
        ret = np.mean(run_policy(env_fn(), pol)["rewards"])
        print(f"{name:26s} {ret:.4f}")


if __name__ == "__main__":
    main()
