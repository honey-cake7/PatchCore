"""Is a trained PPO policy saturated? (diagnostic, no training)

Runs each checkpoint for one episode on the cached stream, deterministically
(mean action, as the benchmark does) and sampled (as training does), and
reports per action dim the raw pre-tanh mean (mean / min / max), the fraction
of steps with |mu| > 3 (tanh within 0.5% of ±1: every sample decodes the same),
log_std, admits/evicts per step, and which bank entries get evicted — the
newest (static + one churning slot) or the oldest.

    python bin/diagnose_ppo_policy.py --cache_dir cache/kvasir_polyppvt_staged_abrupt_4 \
        --capacity 2000 --warmup 100 \
        --ppo results/streaming/kvasir_polyppvt_staged_abrupt_4_M2000_iota/ppo_*_s*.pt
"""
import glob
import os
import sys

import click
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from patchcore.streaming.cache import EmbeddingCacheReader
from patchcore.streaming.env import ActionConfig, MemoryMaintenanceEnv
from patchcore.streaming.ppo import load_checkpoint
from patchcore.streaming.teacher import load_teacher

DIMS = ["a0 admit bar/centre", "a1 admit count", "a2 w_age", "a3 w_-hit",
        "a4 w_-redund", "a5 w_dist_recent"]


def run(env, ac, sample, rng):
    env.reset()
    mus, adm, ev, ev_new, ev_old, rew = [], [], [], 0, 0, []
    done = False
    while not done:
        obs = torch.as_tensor(env.last_obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist, _ = ac(obs)
        mu = dist.mean.squeeze(0).numpy()
        a = dist.sample().squeeze(0).numpy() if sample else mu
        mus.append(mu)
        admit, evict = env.decode_action(a)
        if len(evict):
            slots = env.bank.active_slots()
            ins = env.bank.insert_step[slots]
            ev_new += int(np.isin(evict, slots[ins == ins.max()]).sum())
            ev_old += int(np.isin(evict, slots[ins == ins.min()]).sum())
        _, r, done, info = env.step_with_decision(admit, evict)
        adm.append(info["n_admit"]); ev.append(info["n_evict"]); rew.append(r)
    return np.asarray(mus), np.asarray(adm), np.asarray(ev), ev_new, ev_old, float(np.mean(rew))


@click.command()
@click.option("--cache_dir", required=True)
@click.option("--capacity", type=int, required=True)
@click.option("--warmup", type=int, default=100)
@click.option("--ppo", "ppo_glob", required=True, help="checkpoint path or glob")
@click.option("--teacher", default=None, help="default: <cache_dir>/teacher_k5.npz")
@click.option("--heldout_seed", type=int, default=12345,
              help="distinct from every training/eval seed so H is out-of-sample")
def main(cache_dir, capacity, warmup, ppo_glob, teacher, heldout_seed):
    stream = EmbeddingCacheReader(os.path.join(cache_dir, "stream"))
    teach = load_teacher(teacher or os.path.join(cache_dir, "teacher_k5.npz"))
    for path in sorted(glob.glob(ppo_glob)):
        ac, norm, meta = load_checkpoint(path, device="cpu")
        meta = meta or {}
        print(f"\n=== {path}")
        print(f"action_mode={meta.get('action_mode')} reward_mode={meta.get('reward_mode')} "
              f"kappa={meta.get('admit_cap_abs')} hidden={meta.get('hidden')} "
              f"log_std={np.round(ac.log_std.detach().numpy(), 3).tolist()}")
        for sample in (False, True):
            env = MemoryMaintenanceEnv(
                stream, capacity=capacity, warmup_images=warmup, obs_norm=norm,
                action_cfg=ActionConfig(mode=meta.get("action_mode", "continuous6")),
                reward_mode="iota", teacher=teach,
                heldout_frac=meta.get("heldout_frac") or 0.05, heldout_seed=heldout_seed,
                admit_cap_abs=meta.get("admit_cap_abs"))
            mus, adm, ev, ev_new, ev_old, ret = run(env, ac, sample, None)
            tag = "sampled" if sample else "deterministic"
            print(f"  [{tag}] iota return {ret:.4f}  admits/step {adm.mean():.2f} "
                  f"(zero-admit steps {np.mean(adm == 0):.0%})  evicts/step {ev.mean():.2f}  "
                  f"evicted newest {ev_new / max(ev.sum(), 1):.0%} / oldest {ev_old / max(ev.sum(), 1):.0%}")
            if not sample:
                for d, name in enumerate(DIMS):
                    m = mus[:, d]
                    print(f"    {name:18s} mu mean {m.mean():+8.2f}  min {m.min():+8.2f}  "
                          f"max {m.max():+8.2f}  |mu|>3 {np.mean(np.abs(m) > 3):5.0%}  "
                          f"tanh mean {np.tanh(m).mean():+.3f}")


if __name__ == "__main__":
    main()
