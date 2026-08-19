# Stage-0 reward target: before / after

**Date:** 2026-08-04 · **Config:** M=2000 fixed, WARMUP=15 (MVTec) / 100 (HyperKvasir),
`PERMUTE=200`, 3 eval seeds · **Run:** `STAGE0_WEIGHT=1 DRIFTED_WEIGHT=0 FORGET_WEIGHT=0`

## What changed

The proxy reward's weights are grid-fitted so that mean episode reward Spearman-ranks
policies like labeled AUROC. That ranking target used to be

```
mean(stage_aurocs[s] for s >= 1) + 0.5 * forget_auroc      # stage 0 filtered out
```

which **excludes stage 0** — the only column the paper reports. It is now a weighted
composition (`experiments.fit_reward_weights`, defaults reproduce the old formula):

```
drifted_weight * mean(drifted) + forget_weight * forgetting + stage0_weight * stage0
```

Stage-0 AUROC was already stored in every trace, so re-targeting is an offline
`--traces_in` refit — seconds per class, no re-recording.

Also added: `--permute N`, a permutation null that refits the same 162k-candidate grid
against shuffled targets. It exists because the grid reaches ρ ≈ 0.8 on **pure noise**
with only 12 trace points, so `MIN_RHO=0.7` cannot separate a real fit from search
capacity. Every fit below now carries a p-value.

---

## The win: HyperKvasir (controlled A/B)

Same capacity (M=2000), same warmup, same LR schedule, same step budget, same backbone
(polyp-pvt). **Only the reward target changed.**

| | before (drifted+forget) | after (stage-0) |
|---|---|---|
| reward fit ρ | 0.734 (ceiling ~0.74 after a forget_weight sweep) | **0.937** |
| fit significance | not measured | **p = 0.000** |
| image AUROC (stage mean) | 0.834 — **last of all policies** | **0.870 — 2nd**, level with fifo 0.872 |
| forgetting | 0.840 (static 0.904) | **0.905 — best of all policies** (fifo 0.846, periodic 0.852) |
| PRO | — | **best at every stage** (0.564 / 0.565 / 0.501 / 0.421) |
| seed stability | unstable: admits 13.4k vs 3.3k | **exact**: all 3 seeds admit 3,282 |

Stage-0 image AUROC after: ppo 0.905, vs static 0.904, fifo 0.909, periodic 0.910 — a
statistical tie with the best heuristics, achieved with **3,282 memory writes vs fifo
25,464, periodic 126,000, streaming-greedy 315,654** (8×–96× fewer).

This is the thesis in one class: **equal accuracy, better localization and retention, at
a fraction of the maintenance cost** — and the proxy that produces it is now
statistically validated rather than at a ρ ≈ 0.74 ceiling.

---

## MVTec, M=2000 — stage-0 column

| policy | stage-0 image AUROC |
|---|---|
| periodic_coreset | 0.969 |
| fifo | 0.968 |
| **ppo** | **0.955** |
| static | 0.940 |
| streaming_greedy_coreset | 0.935 |
| reservoir | 0.917 |

Re-targeting did **not** close the gap to the heuristics here. What it did do is
eliminate two competing explanations and isolate the real one.

### Screw: the diagnostic case

| | value |
|---|---|
| stage-0 fit | ρ = 0.993, **p = 0.000** (was **noise**, p = 0.195, under the old target) |
| proxy return | ppo **−2.082** — beats periodic −2.172 and fifo −2.199 |
| stage-0 AUROC | ppo **0.752** — loses to periodic 0.883 |

Valid target + statistically real fit + PPO wins the reward + PPO loses the metric.
Transistor and zipper show the same shape (top the proxy, lose stage-0 to fifo).

**Conclusion: off-manifold reward hacking.** The proxy is fitted and validated on the
12 heuristic traces; PPO finds policies *outside* that region which score better on the
proxy and worse on AUROC. This is what `ITERATE=1` (adding trained-PPO traces to the
fitting set) is built to close, and it is now a motivated next experiment rather than a
guess.

### Regime split

HyperKvasir streams ~3,184 steps (48 PPO iterations, best_window 12) and works.
MVTec streams ~200 steps and reward-hacks. Stream length, not capacity, separates them.

---

---

## ITERATE=1 round (stage-0 target + trained-PPO traces): a regression

Same config plus `ITERATE=1`, which appends each class's own trained-PPO traces
(`ppo_r1`, 12 → 14 traces) to the fitting set. Intended to close the off-manifold
reward hacking screw demonstrated. **It made things worse on both datasets.**

| stage-0 | stage-0 target | + ITERATE | Δ |
|---|---|---|---|
| MVTec image AUROC | 0.955 | **0.944** | −0.011 |
| MVTec pixel AUROC | 0.970 | 0.967 | −0.003 |
| MVTec PRO | 0.886 | 0.877 | −0.009 |
| HyperKvasir image AUROC | 0.905 | 0.902 | −0.003 |
| HyperKvasir PRO | **0.564 (1st)** | 0.528 (5th) | −0.036 |
| HyperKvasir admits | **3,282** | **318,201** | ~97× more writes |

The MVTec regression is concentrated in **screw: 0.752 → 0.655** (−0.097); most other
classes are flat or marginally up (metal_nut 0.975 → 0.978, transistor 0.984 → 0.987).

**The HyperKvasir win was destroyed.** The policy went from the most parsimonious
(3,282 admits, best PRO, best forgetting) to the most write-heavy of any policy
(318,201 admits — more than streaming-greedy's 315,654), and PRO fell from 1st to 5th.
It also stopped winning its own reward: proxy −5.139 vs static −5.072, fifo −5.082,
periodic −5.034.

Mechanism: adding two PPO traces to a 12-trace fitting set moves the fitted weights
toward whatever the *previous* policy did, and with n=14 those two points carry ~14% of
the ranking signal. On hyperkvasir the refit flipped `rho_ranking (current weights)`
from −0.021 to −0.055 and pushed the optimum to a weight set that rewards churn.

**Verdict: do not use ITERATE=1 at this trace count.** The run-A (no-iterate) policies
are the ones to keep. If iterated refit is revisited, it needs many more traces per
round so a single policy's behavior cannot dominate the fit.

Two fits also degenerated under the larger trace set: bottle ρ 0.447 (p = 0.560, noise;
was fine before) and carpet ρ 0.196 (p = 0.980). Both trained anyway.

---

## THE comparison: ours (ppo, stage 0) vs the baseline (stock PatchCore)

Baseline = `results/MVTecAD_Results/IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1`,
instance_auroc, mean **0.9907** (0.9902 over the 14 classes that have a stage-0 eval —
toothbrush's warmup consumes stage 0, so it is excluded from both sides).

| run | budget | ppo stage-0 | baseline | **gap** |
|---|---|---|---|---|
| `_m2.5` | 2.5% matched | 0.949 | 0.990 | −0.041 |
| **`_m10`** | **10% matched** | **0.967** | 0.990 | **−0.023** |
| `_m10`, seed 0 | 10% matched | 0.982 | 0.991 | **−0.008** |
| M=2000 | fixed (not matched) | 0.955 | 0.990 | −0.035 |
| M=2000 + ITERATE | fixed (not matched) | 0.944 | 0.990 | −0.046 |

**Only `_m10` is a fair comparison** — stock gets a 10% coreset (M ≈ 16k–31k patches per
class); the M=2000 runs give the streaming bank 8–15× less memory, so their gap to stock
is mostly budget, not method.

Per-class at `_m10` (3-seed), ours vs baseline: bottle/hazelnut/leather 0.000,
tile **+0.004**, carpet −0.001, metal_nut −0.001, wood −0.004, transistor −0.010,
zipper −0.011, grid −0.014, pill −0.017, cable −0.023, capsule −0.062,
**screw −0.184**. Ten of fourteen are within ~1 pt; the mean gap is carried by screw.

## Budget context (earlier runs, old target)

| run | budget | ppo stage-0 (3-seed) |
|---|---|---|
| `_m2.5` | 2.5% of stream patches (M ≈ 4.1k–7.7k) | 0.949 |
| `_m10` | 10% (M ≈ 16k–31k) | 0.967 |
| this run | fixed M = 2000 | 0.955 |

Suggestive but **not controlled**: M=2000 is a tighter budget than either matched run,
yet the stage-0-targeted policy scores above the 2.5% run which had 2–4× more memory.
Attributing that to the target change requires an M=2000 run under the old target with
its stage-0 column, which is not on hand.

For reference, stock PatchCore (WRN50, 10% coreset, `IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1`)
has image mean 0.9907 and pixel mean 0.9811.

---

## Validity notes — read before quoting any ρ

- **The 162k-candidate grid reaches ρ ≈ 0.8 on shuffled targets** with 12 trace points and
  no held-out split. Fitted ρ is in-sample. Always cite the permutation p-value.
- `MIN_RHO=0.7` sits **inside** the noise band. The null-calibrated gate is `--max_p 0.05`
  (implemented, not yet wired into `run_streaming.sh`).
- **Still noise under the stage-0 target:** zipper (p = 0.235, also failed `min_rho` at
  0.691 and trained anyway), tile (p = 0.080).
- **Toothbrush's row in this run is invalid.** Its warmup consumes stage 0, so a
  stage-0-only target was constant across all traces → the fit crashed with `IndexError`
  → no JSON was written → `run_streaming.sh` does not check step 3.5's exit code → step 4
  silently loaded a **stale** `reward_weights.json` from an earlier experiment. Now fixed
  to raise a named `ValueError`; the exit-code check in `run_streaming.sh` is still open.
- After an `ITERATE=1` round the trace pkl permanently gains `ppo_r*` entries, so later
  p-values describe a larger fitting set and are not directly comparable to these.

## Reproduce

```bash
# this run
STAGE0_WEIGHT=1 DRIFTED_WEIGHT=0 FORGET_WEIGHT=0 PERMUTE=200 \
    ONLY=mvtec ./submit_all_streaming.sh

# offline: does a class's fitted proxy track stage 0 at all?
python bin/diagnose_reward_target.py --glob 'results/streaming/*_m10'

# offline: is a class's fit distinguishable from noise?
python bin/fit_reward_weights.py --traces_in <dir>/reward_traces.pkl \
    --stage0_weight 1 --drifted_weight 0 --forget_weight 0 \
    --min_rho 0 --permute 200 --out /tmp/w.json
```
