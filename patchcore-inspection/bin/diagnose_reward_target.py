"""Ask whether a class's fitted proxy reward actually tracks stage-0 AUROC.

The reward weights in ``reward_weights.json`` are grid-fitted to rank policies
by ``drifted-stage AUROC + forget_weight * forgetting`` — a target that
excludes stage 0. The paper reports stage 0 only. This script re-scores the
saved traces (no re-recording, no GPU, no embedding cache) and prints, per
class, how well the fitted proxy ranks policies by each target:

    rho_fit    Spearman(proxy, the target the weights were fitted to)
               — self-check: must reproduce the logged rho_ranking
    rho_stage0 Spearman(proxy, stage-0 AUROC)          <- the number that matters
    rho_t_s0   Spearman(fitted target, stage-0 AUROC)  — do the targets agree?

A low or negative rho_stage0 on a class means the reward PPO trained against
carries little information about the metric being reported for it.

Usage:
    python bin/diagnose_reward_target.py results/streaming/*_m10
    python bin/diagnose_reward_target.py --glob 'results/streaming/*_m2.5'
"""
import glob as globlib
import json
import os
import pickle
import sys

import click
import numpy as np
import torch  # noqa: F401 — must load before faiss (via patchcore.common) so a
              # single OpenMP runtime wins; reversed order segfaults on macOS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _proxy(traces, w):
    """Mean per-step negated weighted cost per trace, at weights ``w``.

    Mirrors ``experiments.fit_reward_weights.rho_for`` exactly, including the
    offline re-derivation of churn_excess (raw churn is stored) and of
    Q = C90/C (never stored).
    """
    out = []
    for tr in traces:
        c = tr["C"].mean()
        c90 = tr["C90"].mean() if "C90" in tr else 0.0
        p = tr["P"].mean() if "P" in tr else 0.0
        q = (float(np.mean(tr["C90"] / np.maximum(tr["C"], 1e-8)))
             if "C90" in tr else 0.0)
        excess = np.maximum(tr["churn"] - w["churn_budget"], 0.0).mean()
        out.append(-(w["alpha"] * c
                     + w["beta"] * tr["R"].mean()
                     + w["gamma"] * tr["score_drift"].mean()
                     + w["churn_coef"] * excess
                     + w["c90_coef"] * c90
                     + w["probe_coef"] * p
                     + w["q_coef"] * q))
    return np.asarray(out)


def _targets(traces, drifted_w, forget_w, stage0_w):
    out = []
    for tr in traces:
        drifted = [v for s, v in tr["stage_aurocs"].items() if s >= 1]
        t = drifted_w * float(np.mean(drifted)) if drifted else 0.0
        t += forget_w * float(tr["forget_auroc"])
        if stage0_w:
            s0 = tr["stage_aurocs"].get(0)
            t += stage0_w * float(s0) if s0 is not None else 0.0
        out.append(t)
    return np.asarray(out)


def _stage0(traces):
    """Per-trace stage-0 AUROC, or None if any trace lacks a stage-0 eval."""
    vals = [tr["stage_aurocs"].get(0) for tr in traces]
    if any(v is None for v in vals):
        return None
    return np.asarray([float(v) for v in vals])


@click.command()
@click.argument("result_dirs", nargs=-1)
@click.option("--glob", "glob_pat", default=None,
              help="Shell glob for result dirs (quote it to avoid shell expansion)")
def main(result_dirs, glob_pat):
    from scipy import stats

    dirs = list(result_dirs)
    if glob_pat:
        dirs += sorted(globlib.glob(glob_pat))
    if not dirs:
        raise click.UsageError("pass result dirs or --glob")

    print(f"{'class':34s} {'n':>3s} {'rho_fit':>8s} {'logged':>8s} "
          f"{'rho_stage0':>11s} {'rho_t_s0':>9s}")
    rows = []
    for d in dirs:
        pkl = os.path.join(d, "reward_traces.pkl")
        wj = os.path.join(d, "reward_weights.json")
        if not (os.path.exists(pkl) and os.path.exists(wj)):
            print(f"{os.path.basename(d)[:34]:34s} {'--':>3s} "
                  "(missing reward_traces.pkl or reward_weights.json)")
            continue
        with open(pkl, "rb") as f:
            traces = pickle.load(f)
        with open(wj) as f:
            res = json.load(f)

        w = res["recommended"]
        # the target these weights were actually fitted to (older JSONs predate
        # the drifted/stage0 knobs and used the historical target)
        fw = float(res.get("forget_weight", 0.5))
        dw = float(res.get("drifted_weight", 1.0))
        sw = float(res.get("stage0_weight", 0.0))

        proxy = _proxy(traces, w)
        fit_t = _targets(traces, dw, fw, sw)
        s0 = _stage0(traces)

        rho_fit = float(stats.spearmanr(proxy, fit_t).correlation)
        logged = float(res.get("rho_ranking", float("nan")))
        if s0 is None:
            rho_s0 = rho_ts0 = float("nan")
        else:
            rho_s0 = float(stats.spearmanr(proxy, s0).correlation)
            rho_ts0 = float(stats.spearmanr(fit_t, s0).correlation)

        name = os.path.basename(os.path.normpath(d))
        flag = "" if abs(rho_fit - logged) < 5e-3 else "  <- self-check MISMATCH"
        print(f"{name[:34]:34s} {len(traces):3d} {rho_fit:8.3f} {logged:8.3f} "
              f"{rho_s0:11.3f} {rho_ts0:9.3f}{flag}")
        rows.append((rho_fit, logged, rho_s0, rho_ts0))

    if rows:
        arr = np.asarray(rows, dtype=float)
        with np.errstate(invalid="ignore"):
            print(f"{'MEAN':34s} {len(rows):3d} {np.nanmean(arr[:, 0]):8.3f} "
                  f"{np.nanmean(arr[:, 1]):8.3f} {np.nanmean(arr[:, 2]):11.3f} "
                  f"{np.nanmean(arr[:, 3]):9.3f}")
        n_nan = int(np.isnan(arr[:, 2]).sum())
        if n_nan:
            print(f"({n_nan} class(es) have no stage-0 eval — nan above)")


if __name__ == "__main__":
    main()
