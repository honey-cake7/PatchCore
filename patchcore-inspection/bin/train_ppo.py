"""Train the self-contained PPO maintenance policy.

Trains on drift trajectories/seeds and evaluates the learned policy's return
against the hand-designed baselines. ``--reward_mode iota`` (primary) trains
on the label-free normal-score-inflation reward against a teacher table built
by bin/build_teacher.py; ``proxy`` is the legacy weighted proxy. With ``--synthetic`` it runs fully
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


# Synthetic streams have no teacher file; the in-process teacher uses the
# k that run_streaming.sh evaluates with (--n_nn 5).
_SYNTH_TEACHER_K = 5


def _make_synthetic_env_fns(n_env, capacity, warmup, action_mode, base_seed, reward_cfg,
                            env_kw):
    from patchcore.streaming.synthetic import SyntheticConfig, make_synthetic_stream
    from patchcore.streaming.teacher import build_teacher

    fns = []
    for i in range(n_env):
        seed = base_seed + i
        cfg = SyntheticConfig(seed=seed)
        stream = make_synthetic_stream(cfg)
        kw = dict(env_kw)
        if kw.get("reward_mode") == "iota":
            # per-seed streams differ, so each needs its own teacher
            kw["teacher"] = build_teacher(stream, _SYNTH_TEACHER_K, verbose=False)

        def fn(stream=stream, seed=seed, kw=kw):
            return MemoryMaintenanceEnv(
                stream, capacity=capacity, warmup_images=warmup, seed=seed,
                action_cfg=ActionConfig(mode=action_mode),
                # each env gets its own copy: reset() writes warmup scales into it
                reward_cfg=dataclasses.replace(reward_cfg), **kw,
            )

        fns.append(fn)
    return fns


def _make_cache_env_fns(n_env, cache_dir, capacity, warmup, action_mode, base_seed, reward_cfg,
                        env_kw, preshift=False):
    from patchcore.streaming.cache import EmbeddingCacheReader, stage0_end

    reader = EmbeddingCacheReader(os.path.join(cache_dir, "stream"))
    # Pre-shift-only: every training episode (and the obs-norm prefit, which
    # builds its env from these fns) ends before the first stage-1 image.
    stop = stage0_end(reader) if preshift else None
    if preshift:
        print(f"[preshift] training episodes end at image {stop} (first stage-1 image)")

    # Every env replays the same stream, so the expensive warmup coreset (and
    # probe/projection/reward scales) is computed once on a prototype and
    # shared; each env keeps its own seed for per-step randomness.
    print("[env] computing warmup coreset once (shared across envs) ...")
    proto = MemoryMaintenanceEnv(
        reader, capacity=capacity, warmup_images=warmup, seed=base_seed,
        action_cfg=ActionConfig(mode=action_mode),
        reward_cfg=dataclasses.replace(reward_cfg), episode_stop=stop, **env_kw,
    )
    proto.reset()
    init_state = proto.export_init_state()

    fns = []
    for i in range(n_env):
        def fn(seed=base_seed + i):
            env = MemoryMaintenanceEnv(
                reader, capacity=capacity, warmup_images=warmup, seed=seed,
                action_cfg=ActionConfig(mode=action_mode),
                reward_cfg=dataclasses.replace(reward_cfg), episode_stop=stop,
                **env_kw,
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
              type=click.Choice(["continuous6", "band6", "bar_only", "discrete4"]))
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
@click.option("--preshift", is_flag=True, default=False,
              help="Train (and fit obs normalization) on the pre-shift stream "
                   "only: episodes end at the first stage-1 image")
@click.option("--reward_mode", default="proxy", type=click.Choice(["proxy", "iota"]),
              help="iota = label-free normal-score inflation vs a full-memory "
                   "teacher (primary); proxy = legacy weighted proxy")
@click.option("--teacher", "teacher_path", default=None,
              help="Teacher table (bin/build_teacher.py output); required for "
                   "--reward_mode iota with --cache_dir")
@click.option("--heldout_frac", type=float, default=None,
              help="Whole-image held-out fraction withheld from admission "
                   "(default 0.05 for iota, 0 for proxy)")
@click.option("--admit_cap_abs", type=int, default=None,
              help="kappa: max patches admitted per image, every policy (default unlimited)")
@click.option("--adv_baseline", default=None, type=click.Choice(["mean", "loo"]),
              help="grpo baseline: group mean incl. self or leave-one-out "
                   "(default loo for iota, mean for proxy)")
@click.option("--hidden", type=int, default=128, help="Actor/critic MLP width")
@click.option("--epochs", type=int, default=4, help="PPO epochs per iteration")
@click.option("--minibatches", type=int, default=64, help="PPO minibatches per epoch")
def main(cache_dir, synthetic, n_env, capacity, warmup, action_mode,
         total_env_steps, seed, out, device, eval_baselines, reward_json,
         beta, gamma, churn_coef, churn_budget, reward_form, norm_mode,
         reward_scale, adv_mode, clip_mode, clip_high, ent_coef, ent_coef_end,
         lr, lr_end, preshift, reward_mode, teacher_path, heldout_frac,
         admit_cap_abs, adv_baseline, hidden, epochs, minibatches):
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
    if heldout_frac is None:
        heldout_frac = 0.05 if reward_mode == "iota" else 0.0
    if adv_baseline is None:
        adv_baseline = "loo" if reward_mode == "iota" else "mean"
    # heldout_seed = --seed for every env: the lockstep group shares H.
    env_kw = dict(reward_mode=reward_mode, heldout_frac=heldout_frac,
                  heldout_seed=seed, admit_cap_abs=admit_cap_abs)
    if reward_mode == "iota" and not synthetic:
        if not teacher_path:
            raise click.UsageError("--reward_mode iota needs --teacher (bin/build_teacher.py)")
        from patchcore.streaming.teacher import load_teacher

        env_kw["teacher"] = load_teacher(teacher_path)  # loaded once, shared
        print(f"[reward] iota: teacher {teacher_path} (k={env_kw['teacher']['k']}, "
              f"eps={env_kw['teacher']['eps']:.4g})")
    print(f"[env] reward_mode={reward_mode} heldout_frac={heldout_frac} "
          f"admit_cap_abs={admit_cap_abs}")
    if synthetic:
        env_fns = _make_synthetic_env_fns(
            n_env, capacity, warmup, action_mode, seed, reward_cfg, env_kw)
    elif cache_dir:
        env_fns = _make_cache_env_fns(
            n_env, cache_dir, capacity, warmup, action_mode, seed, reward_cfg,
            env_kw, preshift=preshift)
    else:
        raise click.UsageError("pass --cache_dir or --synthetic")

    obs_norm = _fit_obs_norm(env_fns[0], seed=seed) if norm_mode == "episode" else None

    cfg = PPOConfig(
        total_env_steps=total_env_steps, device=device,
        reward_scale=reward_scale, adv_mode=adv_mode,
        clip_mode=clip_mode, clip_high=clip_high,
        ent_coef=ent_coef, ent_coef_end=ent_coef_end, lr=lr, lr_end=lr_end,
        adv_baseline=adv_baseline, hidden=hidden, epochs=epochs,
        minibatches=minibatches,
    )
    if cache_dir:
        # The best-checkpoint score is a rolling mean over one full stream
        # replay; the window must track this dataset's episode length or the
        # selection inherits stream-phase bias (default 9 fits only kvasir).
        # Floor of 3: streams shorter than a rollout (small MVTec classes)
        # would otherwise select on a single noisy iteration.
        from patchcore.streaming.cache import EmbeddingCacheReader, stage0_end

        sr = EmbeddingCacheReader(os.path.join(cache_dir, "stream"))
        episode_steps = (stage0_end(sr) if preshift else sr.n_images) - warmup
        if episode_steps <= 0:
            raise click.UsageError(
                f"no training steps: warmup {warmup} >= episode end "
                f"{episode_steps + warmup}" + (" (pre-shift)" if preshift else ""))
        cfg.best_window = max(3, round(episode_steps / cfg.rollout_steps))
    print(f"[ppo] adv_mode={adv_mode} adv_baseline={adv_baseline} clip_mode={clip_mode} "
          f"clip_high={clip_high} best_window={cfg.best_window} hidden={hidden} "
          f"epochs={epochs} minibatches={minibatches}")
    # Cache-path envs replay the same stream in lockstep -> fused cross-env
    # k-NN batching in collect(); synthetic envs have per-seed streams.
    trainer = PPOTrainer(env_fns, cfg, obs_norm=obs_norm, lockstep=bool(cache_dir))
    history = trainer.train()
    trainer.save(out)
    print(f"saved policy -> {out}")

    if eval_baselines:
        _compare_to_baselines(trainer, env_fns[0])


def _compare_to_baselines(trainer, env_fn):
    """Report mean return of PPO vs each baseline on a fresh env (same reward
    mode / held-out set / kappa as training, via env_fn)."""
    from patchcore.streaming import policies as P
    from patchcore.streaming.policies import PPOPolicy, run_policy

    print(f"\n=== {trainer.envs[0].reward_mode} return (higher is better) ===")
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
