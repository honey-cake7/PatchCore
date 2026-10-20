"""Can a policy learn from the ι reward on this stream? (diagnostic, no training)

For several branch points t across the stream (3 per phase), from the SAME env
state (identical fifo history), branch one decision:
  A  admit the 64 most novel patches, evicting the most redundant
  B  no-op
then roll both forward H steps under the static policy, so the ONLY difference
is the single decision at t. Reports the return difference and compares it with
the noise PPO's group baseline sees: the std across 8 envs of the H-step return
under a random policy from the same state. SNR = |A-B| / noise.

Also checks the incremental ι tracker against a full recompute at each branch
point (a bug would show up as a mismatch).

    python bin/diagnose_iota_signal.py --cache_dir cache/kvasir_polyppvt_staged_abrupt_4 \
        --capacity 2000 --warmup 100
"""
import os
import sys

import click
import torch  # noqa: F401 — load before faiss (macOS libomp)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from patchcore.streaming import policies as P
from patchcore.streaming.cache import EmbeddingCacheReader
from patchcore.streaming.env import MemoryMaintenanceEnv
from patchcore.streaming.reward import iota as iota_fn
from patchcore.streaming.teacher import load_teacher


@click.command()
@click.option("--cache_dir", required=True)
@click.option("--capacity", type=int, required=True)
@click.option("--warmup", type=int, default=100)
@click.option("--teacher", default=None, help="default: <cache_dir>/teacher_k5.npz")
@click.option("--heldout_frac", type=float, default=0.05)
@click.option("--horizons", default="1,10,200")
@click.option("--n_noise_envs", type=int, default=8)
def main(cache_dir, capacity, warmup, teacher, heldout_frac, horizons, n_noise_envs):
    stream = EmbeddingCacheReader(os.path.join(cache_dir, "stream"))
    teach = load_teacher(teacher or os.path.join(cache_dir, "teacher_k5.npz"))
    Hs = [int(h) for h in horizons.split(",")]

    def make(seed=0):
        return MemoryMaintenanceEnv(stream, capacity=capacity, warmup_images=warmup,
                                    seed=seed, reward_mode="iota", teacher=teach,
                                    heldout_frac=heldout_frac, heldout_seed=0)

    def run_until(env, t_stop):
        env.reset()
        fifo = P.FIFOPolicy(k=8)
        while env._t < t_stop:
            fifo.step(env)

    def forward(env, pol, H):
        tot = 0.0
        for _ in range(H):
            _, r, d, _ = pol.step(env)
            tot += r
            if d:
                break
        return tot

    stages = np.asarray([stream.stage_of(i) for i in range(stream.n_images)])
    points = []
    for s in np.unique(stages):
        ids = np.flatnonzero(stages == s)
        ids = ids[ids > warmup + 1]
        if len(ids) > 4:
            points += [(int(s), int(ids[int(f * (len(ids) - 1))])) for f in (0.2, 0.5, 0.8)]

    print(f"M={capacity} warmup={warmup} heldout_frac={heldout_frac} "
          f"stream={stream.n_images} images, branch points per phase at 20/50/80%")
    print(f"{'phase':>5} {'t':>5} {'H':>4} {'A-B':>10} {'noise':>9} {'SNR':>6}  tracker")
    for s, t0 in points:
        for H in Hs:
            res = {}
            for branch in ("A", "B"):
                env = make()
                run_until(env, t0)
                if branch == "A":
                    adm = np.argsort(-env.batch_novelty)[:64]
                    free = env.capacity - len(env.bank)
                    ev = env._select_evictions(max(0, len(adm) - free), None)
                    _, r0, d, _ = env.step_with_decision(adm, ev)
                else:
                    _, r0, d, _ = env.step_with_decision(np.empty(0, np.int64),
                                                         np.empty(0, np.int64))
                # after a step env._t already points at the NEXT image; if that one
                # is held out it is "seen" but has no teacher score yet (S_F[n, t]
                # is NaN for t < n) — the reward just computed excludes it, so must we
                seen = env.heldout_seen_ids()
                sel = seen <= env._t - 1
                ok = "n/a"
                if sel.any():
                    full = iota_fn(env.heldout_image_scores()[sel],
                                   teach["S_F"][seen[sel], env._t - 1], teach["eps"])
                    ok = "OK" if abs(full + r0) < 1e-4 else f"MISMATCH {full:+.5f} vs {-r0:+.5f}"
                res[branch] = r0 + (0.0 if d else forward(env, P.StaticPolicy(), H - 1))
                res[branch + "_ok"] = ok
            rets = []
            for k in range(n_noise_envs):
                env = make(seed=k)
                run_until(env, t0)
                rets.append(forward(env, P.RandomPolicy(k=8, seed=k), H))
            noise = float(np.std(rets))
            diff = res["A"] - res["B"]
            print(f"{s:>5} {t0:>5} {H:>4} {diff:>+10.5f} {noise:>9.5f} "
                  f"{abs(diff) / max(noise, 1e-12):>6.2f}  {res['A_ok']}/{res['B_ok']}")


if __name__ == "__main__":
    main()
