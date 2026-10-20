"""Harvest every number the reviewer checklist needs that EXISTING runs already hold.

No GPU, no re-runs: reads result dirs, cache manifests, reward fits,
checkpoints, SLURM logs and the dataset folders, then writes one CSV per
checklist item plus GAPS.md saying, item by item, what is covered and what
still needs a new run. Run from patchcore-inspection/ on the cluster:

    python bin/harvest_existing_results.py
    python bin/harvest_existing_results.py --polyp_pvt_train_dir /path/to/Polyp-PVT/TrainDataset/image

Outputs (default results/harvest/):
    runs.csv              one row per result dir (budget, warmup, target, job)
    per_seed.csv          every (run, policy, seed, stage) row + writes      [1,2,3]
    memory_sweep.csv      stage-0 mean/std per (run, policy) and budget      [1]
    stock_patchcore.csv   offline PatchCore results found under models/      [1]
    seed_variance.csv     per-class per-seed stage-0 image/pixel AUROC       [2]
    fullstream.csv        AUROC at each quarter of the stream                [3]
    drift_protocol.csv    phases: what changes, when, images per phase       [4]
    preshift_audit.csv    what each run's calibration/training saw           [4]
    calibration.csv       J, traces, rho*, p, B, null p95, pass, w*          [5]
    overlap_*.csv         duplicate-image checks between datasets            [6,8]
    deployment_memory.csv analytic memory per component                      [7]
    protocol.csv          train/test counts, anomaly types, warmup, eval pt  [8]
    hyperparams.json      code defaults + per-run training facts             [8]
    GAPS.md               coverage report

ι-reward protocol (run tags _<budget>_iota[_k<κ>][_pre], see parse_variant),
one CSV per report section [A]-[H], each the full-report table:
    iota_validation.csv   Spearman of -ι vs stage-0 image AUROC          [A]
    kappa_pareto.csv      stage-0 AUROC vs writes per admit cap κ        [B]
    train_seed_variance.csv  PPO std across training vs eval seeds       [C]
    anytime.csv           stage-0 AUROC and ι at every stream fraction   [D]
    less_is_more.csv      full-memory / offline vs streaming at stage 0  [E]
    rl_vs_myopic.csv      ppo vs myopic_iota vs kcenter per budget       [F]
    deployment_measured.csv  deployment.csv of every run (latency, peak) [G]
    preshift_pairs.csv    per-phase AUROC of _pre runs vs their twin     [H]
"""
import csv
import glob
import hashlib
import json
import os
import re
import sys
from collections import defaultdict

import torch  # noqa: F401 — load before anything that may pull in faiss (macOS libomp)
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

BB_TAGS = ("wideresnet50", "polyppvt", "pvtv2_b2", "segformer_mit_b3")
RUN_RE = re.compile(
    r"^(?P<cls>.+)_(?P<bb>" + "|".join(BB_TAGS) + r")_(?P<drift>staged_[a-z]+_\d+)(?P<suffix>.*)$"
)
FORGET = "final_forgetting"
MVTEC = {"bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
         "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"}


def dataset_of(cls):
    return "mvtec" if cls in MVTEC else cls


# Run-tag grammar of the ι-reward protocol. A result dir is
#   <class>_<backbone>_<drift><budget-tag><variant>
#   budget-tag = _M<int> (fixed M) | _m<pct> (M matched to pct% of the stream)
#   variant    = _iota[_k<κ>][_pre]      κ = admit cap in patches/image (absent = ∞)
# Any other variant is a legacy proxy-reward run; its trailing _pre still
# means PRESHIFT=1 (submit_reviewer_runs.sh's _M2000_pre).
IOTA_RE = re.compile(r"^_iota(?:_k(?P<kappa>\d+))?(?P<pre>_pre)?$")
INF = "∞"


def parse_variant(variant):
    """(arm, κ, preshift) of a variant string; κ is an int, "∞" or "" (legacy)."""
    m = IOTA_RE.match(variant)
    if m:
        return "iota", int(m.group("kappa")) if m.group("kappa") else INF, bool(m.group("pre"))
    return "legacy", "", variant.endswith("_pre")


def _kappa_key(k):
    return (k == INF, k if isinstance(k, int) else 0)


def fnum(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def mean_std(vals):
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, len(vals)


def write_csv(path, rows, fields=None):
    fields = fields or (list(rows[0].keys()) if rows else ["empty"])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


def rnd(x, n=4):
    return round(x, n) if isinstance(x, float) and np.isfinite(x) else x


# ---------------------------------------------------------------------------
# SLURM logs: the only place admit/evict counts, data paths, match-mode M and
# PPO iteration counts were recorded. Later jobs override earlier ones per
# result dir, matching how results.csv itself is overwritten.
# ---------------------------------------------------------------------------
BANNER_RES = re.compile(r"^ (?:\[(\S+)\] )?cache=(\S+)\s+results=(\S+)")
BANNER_DATA = re.compile(r"^ (?:\[(\S+)\] )?data=(\S+)\s+M=(\d+)\s+k=(\d+)")
MATCH_M = re.compile(r"^\[(\S+)\] CAPACITY_MODE=match -> M=(\d+)")
STEP = re.compile(r"^\[(?:(\S+) )?(\d(?:\.\d)?)/5\]")
BENCH = re.compile(r"^(\S+)\s+seed=(\d+) mean_reward=(\S+) admit=(\d+) evict=(\d+)")
PPO_ITER = re.compile(r"^\[ppo\] iter (\d+)/(\d+) mean_reward=(\S+) collect=([\d.]+)s .*update=([\d.]+)s")
PPO_BEST = re.compile(r"^\[ppo\] restored best policy from iter (\d+)")
PPO_CFG = re.compile(r"^\[ppo\] adv_mode=(\S+) clip_mode=(\S+) clip_high=(\S+) best_window=(\d+)")
GPU = re.compile(r"^CUDA probe OK: (.+)")


def parse_logs(log_dirs):
    files = sorted({os.path.realpath(f) for d in log_dirs
                    for f in glob.glob(os.path.join(d, "streaming_output_*.log"))})
    job_of = lambda p: int(re.findall(r"(\d+)", os.path.basename(p))[-1])  # noqa: E731
    runs = {}  # result-dir basename -> facts from its most recent job
    for path in sorted(files, key=job_of):
        job = job_of(path)
        gpu, cur, step = None, None, None
        pending = {}  # facts for a dir in THIS job; committed per dir as seen
        with open(path, errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if (m := GPU.match(line)):
                    gpu = m.group(1)
                elif (m := BANNER_DATA.match(line)):
                    pending["_data"] = (m.group(2), int(m.group(3)), int(m.group(4)))
                elif (m := BANNER_RES.match(line)):
                    cur = os.path.basename(m.group(3).rstrip("/"))
                    data, M, k = pending.pop("_data", (None, None, None))
                    pending[cur] = {"job": job, "gpu": gpu, "data_path": data,
                                    "capacity_log": M, "n_nn": k, "bench": {},
                                    "ppo_iters": [], "cache_dir": m.group(2)}
                elif cur is None:
                    continue
                elif (m := MATCH_M.match(line)):
                    pending[cur]["capacity_log"] = int(m.group(2))
                elif (m := STEP.match(line)):
                    step = m.group(2)
                elif step == "5" and (m := BENCH.match(line)):
                    pending[cur]["bench"][(m.group(1), int(m.group(2)))] = {
                        "admit": int(m.group(4)), "evict": int(m.group(5)),
                        "mean_reward": fnum(m.group(3))}
                elif (m := PPO_ITER.match(line)):
                    pending[cur]["ppo_iters"].append(
                        (int(m.group(1)), int(m.group(2)), fnum(m.group(3)),
                         float(m.group(4)), float(m.group(5))))
                elif (m := PPO_BEST.match(line)):
                    pending[cur]["ppo_best_iter"] = int(m.group(1))
                elif (m := PPO_CFG.match(line)):
                    pending[cur].update(adv_mode=m.group(1), clip_mode=m.group(2),
                                        clip_high=m.group(3), best_window=int(m.group(4)))
        for d, facts in pending.items():
            if d.startswith("_"):
                continue
            old = runs.get(d, {})
            # a job that died before benchmarking must not erase the writes of
            # the job whose results.csv is still on disk
            if not facts["bench"] and old.get("bench"):
                facts["bench"], facts["bench_job"] = old["bench"], old.get("bench_job")
            elif facts["bench"]:
                facts["bench_job"] = job
            runs[d] = {**old, **{k: v for k, v in facts.items() if v not in (None, [], {})}}
    return runs, len(files)


# ---------------------------------------------------------------------------
def load_cache_facts(cache_dir):
    out = {}
    man = os.path.join(cache_dir, "stream", "manifest.json")
    if not os.path.exists(man):
        return out
    with open(man) as f:
        m = json.load(f)
    stage_ids = m.get("stage_ids", [])
    out.update(n_stream=m.get("n_images"), patches_per_image=m.get("patches_per_image"),
               dim=m.get("dim"), patch_shape=m.get("patch_shape"), backbone=m.get("backbone"),
               layers=m.get("layers"), drift=m.get("drift"), drift_mode=m.get("drift_mode"),
               stream_seed=m.get("seed"), git_sha=m.get("git_sha"),
               stage_counts=[int(c) for c in np.bincount(stage_ids)] if stage_ids else [])
    tests = {}
    for d in sorted(glob.glob(os.path.join(cache_dir, "test", "stage_*"))):
        lab = os.path.join(d, "labels.npy")
        if not os.path.exists(lab):
            continue
        labels = np.load(lab)
        t = {"n": int(len(labels)), "normal": int((labels == 0).sum()),
             "abnormal": int((labels == 1).sum())}
        mp = os.path.join(d, "masks.npy")
        if os.path.exists(mp):
            masks = np.load(mp, mmap_mode="r")
            t["nonempty_masks"] = int(sum(bool(masks[i].any()) for i in range(len(masks))))
        tests[os.path.basename(d)] = t
    out["tests"] = tests
    return out


def discover_runs(results_root, cache_root, log_facts):
    runs = []
    for d in sorted(glob.glob(os.path.join(results_root, "*/"))):
        name = os.path.basename(d.rstrip("/"))
        m = RUN_RE.match(name)
        if not m or not (os.path.exists(os.path.join(d, "results.csv"))
                         or os.path.exists(os.path.join(d, "reward_weights.json"))):
            continue
        r = {"run": name, "dir": d, "class": m.group("cls"), "dataset": dataset_of(m.group("cls")),
             "backbone": m.group("bb"), "drift": m.group("drift"), "suffix": m.group("suffix")}
        cfg = os.path.join(d, "reward_traces.cfg")
        if os.path.exists(cfg):
            parts = open(cfg).read().strip().split(":")
            r["warmup"] = int(parts[0])
            r["capacity"] = int(parts[1])
            r["stage0_only"] = parts[2] == "1" if len(parts) > 2 else False
        lf = log_facts.get(name, {})
        mm = re.match(r"_m([\d.]+)(.*)$", r["suffix"])
        # fixed-M tags carry the budget too (_M2000_rv): strip it so every
        # budget of one protocol shares a variant name
        mf = re.match(r"_M(\d+)(.*)$", r["suffix"])
        r["capacity"] = r.get("capacity") or lf.get("capacity_log") or (int(mf.group(1)) if mf else None)
        r["log"] = lf
        r["job"] = lf.get("bench_job") or lf.get("job")
        r["budget"] = f"{mm.group(1)}% match" if mm else (
            f"M={r['capacity']}" if r.get("capacity") else "M=?")
        r["variant"] = (mm.group(2) if mm else mf.group(2) if mf else r["suffix"]) or "(untagged)"
        r["arm"], r["kappa"], r["preshift"] = parse_variant(r["variant"])
        dp = os.path.join(d, "deployment.csv")
        r["deploy_measured"] = list(csv.DictReader(open(dp))) if os.path.exists(dp) else []
        tag = name[: len(name) - len(r["suffix"])] if r["suffix"] else name
        r["cache_dir"] = os.path.join(cache_root, tag)
        r["cache"] = load_cache_facts(r["cache_dir"])
        wj = os.path.join(d, "reward_weights.json")
        r["fit"] = json.load(open(wj)) if os.path.exists(wj) else None
        ck = glob.glob(os.path.join(d, "ppo_*.pt"))
        r["ckpt"] = ck[0] if ck else None
        runs.append(r)
    return runs


# ---------------------------------------------------------------------------
def harvest_per_seed(runs):
    rows = []
    for r in runs:
        p = os.path.join(r["dir"], "results.csv")
        if not os.path.exists(p):
            continue
        n_stages = len(r["cache"].get("stage_counts") or []) or 4
        bench = r["log"].get("bench", {})
        for row in csv.DictReader(open(p)):
            seed = int(row["seed"]) if row.get("seed") not in ("", None) else 0
            # train_seed: "" for non-PPO rows and for files that predate the column
            ts = row.get("train_seed") or ""
            ts = int(fnum(ts)) if np.isfinite(fnum(ts)) else ""
            if row["stage"] == FORGET:
                frac, test_set = 1.0, "stage_0"
            elif row["stage"].startswith("stage0_at_"):  # stage-0 test mid-stream
                frac, test_set = (int(row["stage"].rsplit("_", 1)[1]) + 1) / n_stages, "stage_0"
            else:
                s = int(row["stage"])
                frac, test_set = (s + 1) / n_stages, f"stage_{s}"
            # writes/evict: results.csv columns when present (exact, per train
            # seed); else the step-5 log line, which cannot tell train seeds apart
            b = bench.get((row["policy"], seed), {})
            w, ev = fnum(row.get("writes")), fnum(row.get("evict"))
            from_csv = np.isfinite(w)
            rows.append({
                "dataset": r["dataset"], "class": r["class"], "backbone": r["backbone"],
                "run": r["run"], "budget": r["budget"], "variant": r["variant"],
                "arm": r["arm"], "kappa": r["kappa"], "preshift": r["preshift"],
                "capacity": r.get("capacity"), "warmup": r.get("warmup"),
                "policy": row["policy"], "seed": seed, "train_seed": ts, "stage": row["stage"],
                "stream_frac_evaluated_at": frac, "test_set": test_set,
                "image_auroc": fnum(row["image_auroc"]), "pixel_auroc": fnum(row["pixel_auroc"]),
                "pro": fnum(row["pro"]), "iota": fnum(row.get("iota")),
                "writes_admit": int(w) if from_csv else b.get("admit", ""),
                "evict": (int(ev) if np.isfinite(ev) else "") if from_csv else b.get("evict", ""),
                "writes_source": "results.csv" if from_csv else ("log" if b else ""),
                "job": r.get("job"),
            })
    return rows


def summarize(rows, keys, stage_filter):
    groups = defaultdict(list)
    for x in rows:
        if stage_filter(x):
            groups[tuple(x[k] for k in keys)].append(x)
    out = []
    for k, g in groups.items():
        rec = dict(zip(keys, k))
        for m in ("image_auroc", "pixel_auroc", "pro"):
            mu, sd, n = mean_std([x[m] for x in g])
            rec[f"{m}_mean"], rec[f"{m}_std"] = rnd(mu), rnd(sd)
        rec["n_seeds"] = len({x["seed"] for x in g})
        w = [float(x["writes_admit"]) for x in g if x["writes_admit"] != ""]
        rec["writes_mean"] = rnd(float(np.mean(w)), 1) if w else ""
        out.append(rec)
    return out


def pooled(per_seed, keys, stage="0"):
    """Cross-class summarize(): each (class, eval seed, train seed) is one sample."""
    out = summarize([{**x, "seed": (x["class"], x["seed"], x["train_seed"])} for x in per_seed],
                    keys, lambda x: x["stage"] == stage)
    for d in out:
        d["n_class_seed_pairs"] = d.pop("n_seeds")
    return out


def harvest_stock(pkg):
    rows = []
    for p in glob.glob(os.path.join(pkg, "models", "*", "results.csv")) + \
            glob.glob(os.path.join(pkg, "results", "*", "*", "results.csv")):
        if "/streaming/" in p:
            continue
        cfg = os.path.basename(os.path.dirname(p))
        pm = re.search(r"_P(\d+)(?:_|$)", cfg)
        pct = {"01": 10.0, "001": 1.0, "0001": 0.1}.get(pm.group(1)) if pm else None
        try:
            reader = csv.DictReader(open(p))
            for row in reader:
                name = row.get("Row Names", "")
                if not name or name.lower().startswith("mean"):
                    continue
                rows.append({"config": cfg, "source": p, "coreset_pct": pct,
                             "class": name.split("_", 1)[-1],
                             "image_auroc": fnum(row.get("instance_auroc")),
                             "pixel_auroc": fnum(row.get("full_pixel_auroc")),
                             "pro": fnum(row.get("full_pro"))})
        except Exception as e:  # stray CSVs with another layout
            print(f"  [stock] skip {p}: {e}")
    return rows


def backfill_null(r, n_perm):
    """Permutation null for a fit recorded with PERMUTE=0, from its saved traces.

    Uses only the traces the fit actually used (keys of fit["targets"]): a later
    ITERATE round may have appended ppo_rN traces to the pkl. Returns ({}, why)
    when it cannot reproduce the fit, so a mismatched null is never reported.
    """
    import pickle

    from patchcore.streaming.experiments import permutation_null

    pkl = os.path.join(r["dir"], "reward_traces.pkl")
    targets = (r["fit"] or {}).get("targets") or {}
    if not os.path.exists(pkl) or not targets:
        return {}, "no reward_traces.pkl / targets"
    traces = [t for t in pickle.load(open(pkl, "rb")) if f"{t['policy']}:{t['seed']}" in targets]
    if len(traces) != len(targets):
        return {}, f"pkl has {len(traces)}/{len(targets)} of the fit's traces"
    null = permutation_null(traces, np.asarray([targets[f"{t['policy']}:{t['seed']}"] for t in traces]),
                            n_permutations=n_perm)
    if abs(null["observed_best_rho"] - r["fit"]["rho_ranking"]) > 5e-3:
        return {}, (f"null observed {null['observed_best_rho']:.3f} != fitted "
                    f"{r['fit']['rho_ranking']:.3f} (fit used a different grid)")
    return null, "backfilled from reward_traces.pkl"


def harvest_calibration(runs, backfill_perm=200):
    rows = []
    for r in runs:
        fit = r["fit"]
        if not fit:
            continue
        pn = fit.get("permutation_null") or {}
        source = "fit JSON" if pn else ""
        if not pn and backfill_perm:
            try:
                pn, source = backfill_null(r, backfill_perm)
            except Exception as e:  # never let one bad pkl stop the harvest
                pn, source = {}, f"backfill failed: {e}"
            print(f"  [calib] {r['run']}: {source}"
                  + (f" -> p={pn['p_value']:.3f}" if pn else ""))
        rec = {"run": r["run"], "dataset": r["dataset"], "class": r["class"],
               "backbone": r["backbone"], "capacity": r.get("capacity"),
               "target": f"{fit.get('drifted_weight', 1.0)}*drifted+"
                         f"{fit.get('forget_weight', 0.5)}*forget+"
                         f"{fit.get('stage0_weight', 0.0)}*stage0",
               "J_n_traces": fit.get("n_traces"),
               "reference_trajectories": " ".join(fit.get("trace_policies", [])),
               "rho_star": rnd(fit.get("rho_ranking")),
               "rho_default_weights": rnd(fit.get("rho_ranking_current_weights")),
               "p_value": rnd(pn.get("p_value")) if pn else "not run (PERMUTE=0)",
               "B_permutations": pn.get("n_permutations", ""),
               "null_mean": rnd(pn.get("null_mean")) if pn else "",
               "null_p95": rnd(pn.get("null_p95")) if pn else "",
               "n_candidates": pn.get("n_candidates", ""),
               "pass_p<=0.05": (pn["p_value"] <= 0.05) if pn else "",
               "pass_rho>=0.7": fit.get("rho_ranking", 0) >= 0.7,
               "null_source": source or "not run (PERMUTE=0)",
               "traces_without_stage0": len(fit.get("traces_without_stage0", [])),
               "label_source": "cached TEST split (test/stage_*) — same images as reported metrics"}
        for k, v in fit.get("recommended", {}).items():
            rec[f"w_{k}"] = v
        rows.append(rec)
    return rows


def harvest_drift_protocol():
    from patchcore.streaming import drift as D

    rows = []
    for name, sched in D.SCHEDULES.items():
        for s, p in enumerate(sched.stages):
            rows.append({"schedule": name, "shape": sched.shape, "phase": s,
                         "when": f"stream positions [{s}/{sched.n_stages}, {s + 1}/{sched.n_stages}) "
                                 "of the shuffled normal stream",
                         "wb_gain": p.wb_gain, "hue_shift_deg": p.hue_shift, "gamma": p.gamma,
                         "blur_sigma": p.blur_sigma, "specular": p.specular,
                         "vignette": p.vignette, "jitter": p.jitter})
    return rows


def harvest_preshift(runs):
    rows = []
    for r in runs:
        fit = r["fit"] or {}
        uses_post = bool(fit.get("drifted_weight", 1.0)) or bool(fit.get("forget_weight", 0.5))
        if r["preshift"]:
            # PRESHIFT=1 (run_streaming.sh): fit + PPO training + obs-norm prefit
            # see the pre-shift stream only; the benchmark still runs all phases
            rows.append({
                "run": r["run"], "class": r["class"],
                "fit_target_uses_postshift_labels": uses_post if fit else "no fit",
                "fit_traces_span": "pre-shift stream only" if fit else "no fit",
                "ppo_training_stream": "pre-shift stream only (PRESHIFT=1)",
                "obs_norm_fitted_on": "pre-shift episode",
                "reward_scales_probe_from": "warmup images only (pre-shift)",
                "verdict": "pre-shift-only (PRESHIFT=1, _pre tag)",
            })
            continue
        rows.append({
            "run": r["run"], "class": r["class"],
            "fit_target_uses_postshift_labels": uses_post if fit else "no fit",
            # Facts from the code, true for every run produced by run_streaming.sh:
            "fit_traces_span": "full stream (all phases): reward components averaged over every step",
            "ppo_training_stream": "full stream, all phases (train_ppo env.reset() with no episode_slice)",
            "obs_norm_fitted_on": "one full random-policy episode incl. drifted phases (norm_mode=episode)",
            "reward_scales_probe_from": "warmup images only (pre-shift)",
            "verdict": "NOT pre-shift-only: training and calibration saw post-shift data",
        })
    return rows


def harvest_deployment(runs, pkg):
    rows = []
    bb_params = {}

    def backbone_params(bb):
        if bb in bb_params:
            return bb_params[bb]
        n = ""
        try:
            if bb == "wideresnet50":
                import torchvision
                net = torchvision.models.wide_resnet50_2(weights=None)
                n = sum(p.numel() for name, p in net.named_parameters()
                        if not name.startswith(("layer4", "fc")))  # PatchCore stops at layer3
            elif bb in ("polyppvt", "pvtv2_b2"):
                env = "POLYP_PVT_WEIGHTS" if bb == "polyppvt" else "PVTV2_B2_WEIGHTS"
                default = os.path.join(pkg, "..", "models",
                                       "PolypPVT.pth" if bb == "polyppvt" else "pvt_v2_b2.pth")
                sd = torch.load(os.environ.get(env, default), map_location="cpu", weights_only=False)
                sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
                n = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))  # full net: upper bound
        except Exception as e:
            print(f"  [deploy] backbone param count for {bb} unavailable: {e}")
        bb_params[bb] = n
        return n

    for r in runs:
        c, M = r["cache"], r.get("capacity")
        if not c or not M:
            continue
        D, P = c["dim"], c["patches_per_image"]
        hold = max(1, int(P * 0.1))  # RewardConfig.holdout_patch_frac
        pol = ""
        if r["ckpt"]:
            try:
                obj = torch.load(r["ckpt"], map_location="cpu", weights_only=False)
                sd = obj["ac"] if isinstance(obj, dict) and "ac" in obj else obj
                pol = sum(v.numel() for k, v in sd.items() if not k.startswith("critic"))
            except Exception as e:
                print(f"  [deploy] {r['ckpt']}: {e}")
        bbp = backbone_params(r["backbone"])
        mb = lambda n_floats: round(n_floats * 4 / 2**20, 3)  # noqa: E731 — float32 MiB
        rec = {"run": r["run"], "backbone": r["backbone"], "M": M, "D": D, "patches_per_image": P,
               "backbone_params": bbp, "backbone_MiB_fp32": mb(bbp) if bbp else "",
               "bank_MiB": mb(M * D), "bank_device_mirror_MiB": mb(M * D),
               "policy_actor_params": pol, "policy_MiB": mb(pol) if pol else "",
               "window_MiB": mb(32 * hold * D),  # ObsConfig.window_images=32 holdout slices
               "probe_Z_MiB": mb(256 * D),       # env.reset(): 256 held-out warmup patches
               "note": "analytic fp32 sizes; peak GPU memory incl. activations NOT measured"}
        iters = r["log"].get("ppo_iters", [])
        if iters:
            # training-time cost only (8 lockstep envs x 256 steps per iter)
            rec["train_ms_per_env_step"] = round(1e3 * np.mean([i[3] for i in iters]) / (8 * 256), 2)
        rows.append(rec)
    return rows


def walk_dataset(root):
    out = {}
    if not root or not os.path.isdir(root):
        return out
    from patchcore.streaming.stream import IMG_EXTS

    for split in ("train", "test", "ground_truth"):
        sd = os.path.join(root, split)
        if not os.path.isdir(sd):
            continue
        for t in sorted(os.listdir(sd)):
            td = os.path.join(sd, t)
            if os.path.isdir(td):
                out[f"{split}/{t}"] = sum(f.lower().endswith(IMG_EXTS) for f in os.listdir(td))
    return out


def harvest_protocol(runs, dataset_root):
    rows = []
    for r in runs:
        c = r["cache"]
        data = r["log"].get("data_path")
        local_root = None
        if data:
            cand = os.path.join(data, r["class"])
            local_root = cand if os.path.isdir(cand) else None
        if local_root is None and dataset_root:
            sub = {"mvtec": "mvtec", "kvasir": "kvasir_patchcore",
                   "hyperkvasir": "hyperkvasir_patchcore"}.get(r["dataset"], r["dataset"])
            cand = os.path.join(dataset_root, sub, r["class"])
            local_root = cand if os.path.isdir(cand) else None
        counts = walk_dataset(local_root)
        t0 = (c.get("tests") or {}).get("stage_0", {})
        sc = c.get("stage_counts") or []
        W = r.get("warmup")
        rows.append({
            "run": r["run"], "dataset": r["dataset"], "class": r["class"],
            "backbone": r["backbone"], "M": r.get("capacity"), "warmup_images": W,
            "n_train_stream": c.get("n_stream"), "images_per_phase": sc,
            "stage0_eval_after_image": sc[0] if sc else "",
            "stage0_policy_steps": (sc[0] - W) if sc and W is not None else "",
            "warmup_covers_stage0": (W >= sc[0]) if sc and W is not None else "",
            "test_normal": t0.get("normal"), "test_abnormal": t0.get("abnormal"),
            "test_nonempty_masks": t0.get("nonempty_masks"),
            "anomaly_types": " ".join(k.split("/", 1)[1] for k in counts
                                      if k.startswith("test/") and k != "test/good"),
            "dataset_folder_counts": json.dumps(counts) if counts else "dataset dir not found",
            "dataset_dir": local_root or "",
        })
    return rows


# ---------------------------------------------------------------------------
# Duplicate-image detection: exact (md5) + perceptual (64-bit dHash). dHash
# hits at distance <= 2 are CANDIDATES — endoscopy frames of one anatomy can
# collide — so the pairs CSV is meant to be eyeballed before quoting a count.
# ---------------------------------------------------------------------------
def image_index(root, label):
    from PIL import Image

    from patchcore.streaming.stream import IMG_EXTS

    paths = sorted(p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
                   if p.lower().endswith(IMG_EXTS) and "ground_truth" not in p
                   and "/masks/" not in p)
    md5s, hashes, keep = [], [], []
    for p in paths:
        try:
            with open(p, "rb") as f:
                md5 = hashlib.md5(f.read()).hexdigest()
            im = Image.open(p)
            im.draft("L", (64, 64))
            a = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
            bits = (a[:, 1:] > a[:, :-1]).flatten()
            hashes.append(np.packbits(bits).view(">u8")[0])
            md5s.append(md5)
            keep.append(p)
        except Exception as e:
            print(f"  [overlap] unreadable {p}: {e}")
    print(f"  [overlap] indexed {len(keep)} images under {label}")
    return keep, md5s, np.asarray(hashes, dtype=np.uint64)


_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def compare_sets(a, b, max_hamming=2):
    pa, ma, ha = a
    pb, mb, hb = b
    by_md5 = defaultdict(list)
    for p, m in zip(pb, mb):
        by_md5[m].append(p)
    stems_b = defaultdict(list)
    for p in pb:
        stems_b[os.path.splitext(os.path.basename(p))[0].lower()].append(p)
    pairs = []
    for i in range(0, len(ha), 256):
        x = ha[i:i + 256, None] ^ hb[None, :]
        dist = _POP[x.view(np.uint8)].reshape(x.shape + (8,)).sum(-1)
        for ii, jj in zip(*np.nonzero(dist <= max_hamming)):
            pairs.append((i + ii, int(jj), int(dist[ii, jj])))
    hit = defaultdict(dict)
    for i, j, d in pairs:
        hit[i].setdefault("dhash", (pb[j], d))
    for i, (p, m) in enumerate(zip(pa, ma)):
        if m in by_md5:
            hit[i]["md5"] = by_md5[m][0]
        st = os.path.splitext(os.path.basename(p))[0].lower()
        # HyperKvasir prep may prefix names; match on suffix too
        if st in stems_b:
            hit[i]["name"] = stems_b[st][0]
    rows = []
    for i, h in sorted(hit.items()):
        rows.append({"image_a": pa[i], "md5_match": h.get("md5", ""),
                     "filename_match": h.get("name", ""),
                     "dhash_match": h["dhash"][0] if "dhash" in h else "",
                     "dhash_dist": h["dhash"][1] if "dhash" in h else ""})
    return rows


def split_of(path):
    m = re.search(r"/(train|test)/([^/]+)/", path)
    return f"{m.group(1)}/{m.group(2)}" if m else "other"


def overlap_summary(rows, paths_a, name):
    """Counts per (split of image A, split of its match in B), per match type."""
    n_a = defaultdict(int)
    for p in paths_a:
        n_a[split_of(p)] += 1
    agg = defaultdict(lambda: defaultdict(int))
    for r in rows:
        match = r["md5_match"] or r["filename_match"] or r["dhash_match"]
        key = (split_of(r["image_a"]), split_of(match))
        agg[key]["any"] += 1
        for t in ("md5_match", "filename_match", "dhash_match"):
            agg[key][t] += bool(r[t])
    return [{"comparison": name, "split_a": sa, "n_a": n_a[sa], "split_b": sb,
             "md5_dupes": c["md5_match"], "filename_dupes": c["filename_match"],
             "dhash_candidates": c["dhash_match"], "any_match": c["any"]}
            for (sa, sb), c in sorted(agg.items())]


# ---------------------------------------------------------------------------
def code_hyperparams():
    import dataclasses

    from patchcore.streaming.env import OBS_DIM, ActionConfig, ObsConfig
    from patchcore.streaming.ppo import ACTION_DIM, PPOConfig
    from patchcore.streaming.reward import RewardConfig

    return {
        "ObsConfig": dataclasses.asdict(ObsConfig()),
        "ActionConfig": dataclasses.asdict(ActionConfig()),
        "RewardConfig_defaults": {k: v for k, v in dataclasses.asdict(RewardConfig()).items()},
        "PPOConfig_defaults": dataclasses.asdict(PPOConfig()),
        "network": {"obs_dim": OBS_DIM, "act_dim": ACTION_DIM, "hidden": 128,
                    "actor": "Linear(obs,128)-Tanh-Linear(128,128)-Tanh-Linear(128,act); "
                             "state-independent log_std init -0.5",
                    "critic": "same trunk shape, separate weights (unused by grpo adv_mode)"},
        "action_decoding (env.decode_action, continuous6)":
            "a=tanh(raw). a0 -> admission bar = quantile of batch NN-1 novelty at (a0+1)/2; "
            "a1 -> admit cap = (a1+1)/2 * 0.05 * M patches/step (most novel kept); "
            "a2..a5 -> eviction utility weights over entry features "
            "[normalized age, -hit rate, -NN-2 redundancy, distance to recent-insertion "
            "centroid] (bank.entry_features), lowest utility evicted; "
            "evictions = max(0, n_admit - free slots)",
        "probe_set_Z": "256 patches drawn from warmup images, held OUT of the initial bank",
        "warmup_bank": "greedy coreset of warmup patches (minus probe) down to M",
        "eval": "k-NN (k=--n_nn, run_streaming.sh N_NN=5) mean squared-L2; image score = max "
                "patch score; bank frozen at each stage boundary",
        "grid_W (fit_reward_weights)": {
            "alpha": [0.0, 0.25, 1.0], "beta": [0.0, 0.15, 0.3, 0.6, 1.0],
            "gamma": [0.0, 0.1, 0.3], "churn_coef": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
            "churn_budget": [0.0, 0.005, 0.01, 0.02], "c90_coef": [0.0, 0.25, 0.5, 1.0, 2.0],
            "probe_coef": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0], "q_coef": [0.0, 1.0, 2.0, 4.0, 8.0],
            "size": 3 * 5 * 3 * 6 * 4 * 5 * 6 * 5},
        "reward_terms_in_code": "7 weighted terms (alpha*C, beta*R, gamma*score_drift, "
                                "churn_coef*excess_churn, c90*C90, probe*P, q*Q) + 1 threshold "
                                "(churn_budget) = the 8 numbers in w*",
        "lr_note": "LR is NOT logged or checkpointed. run_streaming.sh default 1e-3->1e-5; "
                   "submit_all_streaming.sh mvtec block 1e-5->1e-6; check the submit command used.",
    }


# ---------------------------------------------------------------------------
# The ι-reward protocol the paper reports (decided with the user; GAPS.md
# checks each item against what is on disk).
IOTA_BUDGETS = {"mvtec": ("M=1000", "M=2000", "2.5% match", "5% match", "10% match"),
                "kvasir": ("M=1000", "M=2000", "M=4742", "M=10000", "1% match"),
                "hyperkvasir": ("M=1000", "M=2000", "M=4742", "M=10000", "1% match")}
IOTA_KAPPAS = (1, 4, 16, 64, INF)  # swept at M=2000
IOTA_POLICIES = {"static", "fifo", "reservoir", "streaming_greedy_coreset", "periodic_coreset",
                 "kcenter", "myopic_iota", "ppo"}
N_TRAIN_SEEDS, N_EVAL_SEEDS = 3, 5
LEGACY_GAPS_MARK = "Sections 1-8 below: "


def iota_coverage(per_seed, deploy_meas):
    s0 = [x for x in per_seed if x["arm"] == "iota" and x["stage"] == "0"]
    main = defaultdict(list)
    for x in s0:
        if x["kappa"] == INF and not x["preshift"]:
            main[(x["dataset"], x["budget"])].append(x)
    L = ["## 0. ι-reward protocol coverage", "",
         f"Target: ι arm (_iota) at 5 budgets x 3 datasets, κ sweep {{1,4,16,64,∞}} at M=2000, "
         f"{N_TRAIN_SEEDS} train seeds x {N_EVAL_SEEDS} eval seeds, _pre endoscopy runs at M=2000, "
         "deployment.csv at M=2000, offline/full-memory reference rows, myopic_iota + kcenter.", ""]
    for ds, budgets in IOTA_BUDGETS.items():
        L.append(f"- **{ds}**: {sum((ds, b) in main for b in budgets)}/5 budgets with _iota stage-0 results")
        for b in budgets:
            g = main.get((ds, b))
            if not g:
                L.append(f"  - {b}: NEW RUN (no _iota results)")
                continue
            pols = {x["policy"] for x in g}
            miss = sorted(IOTA_POLICIES - pols)
            ppo = [x for x in g if x["policy"] == "ppo"]
            ev = len({x["seed"] for x in ppo})
            ts = len({x["train_seed"] for x in ppo if x["train_seed"] != ""})
            stream = [x for x in g if x["policy"] not in NOT_STREAMING]
            n_iota = sum(np.isfinite(x["iota"]) for x in stream)
            ref = [_abbr(p) for p in NOT_STREAMING if p in pols]
            L.append(f"  - {b} ({len({x['class'] for x in g})} classes): "
                     + ("all 8 policies" if not miss else f"MISSING {', '.join(miss)}")
                     + f"; eval seeds {ev}/{N_EVAL_SEEDS}, train seeds {ts}/{N_TRAIN_SEEDS}"
                     + ("" if ev >= N_EVAL_SEEDS and ts >= N_TRAIN_SEEDS else " -> NEW RUN")
                     + f"; ι logged {n_iota}/{len(stream)} rows"
                     + f"; reference rows: {' '.join(ref) if len(ref) == 4 else (' '.join(ref) or 'none') + ' -> NEW RUN (OFFLINE=1 / --full_memory)'}")
        extra = sorted({b for d, b in main if d == ds} - set(budgets), key=_budget_key)
        if extra:
            L.append(f"  - also present (outside the protocol): {', '.join(extra)}")
    L += ["", "- κ sweep at M=2000 (PPO, non-pre):"]
    for ds in IOTA_BUDGETS:
        have = {x["kappa"] for x in s0 if x["dataset"] == ds and x["budget"] == "M=2000"
                and x["policy"] == "ppo" and not x["preshift"]}
        miss = [str(k) for k in IOTA_KAPPAS if k not in have]
        L.append(f"  - {ds}: " + ("all κ" if not miss else f"MISSING κ={','.join(miss)} -> NEW RUN"))
    L += ["- Pre-shift (_iota_pre at M=2000, endoscopy):"]
    runs_ = {x["run"] for x in per_seed}
    for ds in ("kvasir", "hyperkvasir"):
        pre = sorted({x["run"] for x in s0 if x["dataset"] == ds and x["budget"] == "M=2000"
                      and x["preshift"]})
        twin = sum(r[: -len("_pre")] in runs_ for r in pre)
        L.append(f"  - {ds}: {len(pre)} class runs, {twin} with a full-stream twin"
                 + ("" if pre else " -> NEW RUN"))
    L += ["- Deployment (deployment.csv in an _iota M=2000 run):"]
    for ds in IOTA_BUDGETS:
        d = sorted({x["policy"] for x in deploy_meas if x["dataset"] == ds and x["budget"] == "M=2000"
                    and parse_variant(x["variant"])[0] == "iota"})
        L.append(f"  - {ds}: " + (f"measured for {', '.join(d)}" if d else "NEW RUN (DEPLOY=1)"))
    L.append("")
    return L


def gaps_report(runs, per_seed, calib, overlap, stock, n_logs, deploy_meas=()):
    L = ["# Reviewer checklist — what existing runs cover", "",
         f"Scanned {len(runs)} result dirs and {n_logs} SLURM logs. "
         "COVERED = harvested into the CSVs here; NEW RUN = no existing artifact has it.", ""]
    L += iota_coverage(per_seed, list(deploy_meas))
    L += [LEGACY_GAPS_MARK + "the original reviewer checklist, judged on the legacy "
          "proxy-reward runs (ι-arm items are in section 0).", ""]
    by_ds = defaultdict(lambda: defaultdict(set))  # dataset -> M -> policies
    classes_at = defaultdict(lambda: defaultdict(set))
    for x in per_seed:
        if x["stage"] == "0" and x["arm"] == "legacy":
            key = f'{x["budget"]} {x["variant"]}'
            by_ds[x["dataset"]][key].add(x["policy"])
            classes_at[x["dataset"]][key].add(x["class"])
    wanted = {"static", "fifo", "reservoir", "streaming_greedy_coreset", "periodic_coreset", "ppo"}
    L += ["## 1. Memory sweep", ""]
    for ds, budgets in sorted(by_ds.items()):
        L.append(f"- **{ds}**: {len(budgets)} budget(s) with stage-0 results")
        for M, pols in sorted(budgets.items(), key=lambda kv: _budget_key(kv[0])):
            miss = sorted(wanted - pols)
            L.append(f"  - {M} ({len(classes_at[ds][M])} classes): "
                     + ("all 6 methods" if not miss else f"MISSING {', '.join(miss)}"))
        n_b = len({k.split(" ")[0] + (" match" if "match" in k else "") for k in budgets})
        if n_b < 5:
            L.append(f"  - NEW RUN: needs >=5 budgets, has {n_b}")
    has_writes = sum(1 for x in per_seed if x["writes_admit"] != "")
    off = {x["run"] for x in per_seed if x["policy"].startswith("offline_")}
    L += [f"- Writes: {has_writes}/{len(per_seed)} per-seed rows have admit counts from logs"
          + ("" if has_writes == len(per_seed) else " (rest: log missing/rotated -> rerun step 5)"),
          f"- Offline PatchCore: stock results found for {sorted({s['config'] for s in stock}) or 'none'}"
          " (full train set, NOT stage-0 data, NOT matched M).",
          (f"- Offline greedy coreset at exact M: rows in {len(off)} run(s) (see [E])." if off else
           "- NEW RUN: offline greedy coreset at every exact M (incl. M=2000) — never run."), ""]

    seeds, tseeds = defaultdict(set), defaultdict(set)
    for x in per_seed:
        seeds[x["run"]].add(x["seed"])
        if x["policy"] == "ppo" and x["train_seed"] != "":
            tseeds[x["run"]].add(x["train_seed"])
    ts_runs = [r for r, v in tseeds.items() if len(v) >= 2]
    ns = sorted({len(v) for v in seeds.values()})
    L += ["## 2. Seed variance", "",
          f"- COVERED: per-class per-seed rows in seed_variance.csv; eval seeds per run: {ns}.",
          "- NEW RUN: 5 seeds (EVAL_SEEDS=0,1,2,3,4) — only step 5 needs rerunning." if ns and max(ns) < 5 else "",
          "- What the eval seed controls (from code): env rng only — probe-set Z draw, warmup "
          "coreset random start, per-step holdout split, window subsample, obs projection. "
          "It does NOT change stream order (fixed by the cache SEED=0), baseline policy "
          "sampling (reservoir/periodic/random use a hard-coded seed=0), or PPO weights "
          "(one policy trained with TRAIN_SEEDS=0, deterministic mean action). So ppo std = "
          "eval-env noise only, not training variance.",
          (f"- COVERED for {len(ts_runs)} run(s) with >=2 PPO train seeds: train- vs eval-seed std "
           "in train_seed_variance.csv / [C]." if ts_runs else ""), ""]

    full = sorted({x["run"] for x in per_seed if x["stage"] == "3"})
    mid = sorted({x["run"] for x in per_seed if x["stage"].startswith("stage0_at_")})
    pre_runs = sorted({r["run"] for r in runs if r["preshift"]})
    L += ["## 3. Full-stream behaviour", "",
          f"- COVERED for {len(full)} runs: AUROC at 25/50/75/100% — but each on THAT phase's "
          "drifted test set, plus stage-0 test re-scored at 100% (forgetting). See fullstream.csv.",
          (f"- COVERED for {len(mid)} runs: stage-0 test at 50%/75% (anytime.csv / [D])." if mid else
           "- NEW RUN: stage-0 test at 50% and 75% (never evaluated), and every STAGE0_ONLY run "
           "(only the 25% point exists)."), ""]

    L += ["## 4. Drift", "",
          "- COVERED: phase definitions (drift_protocol.csv), images per phase and per-phase "
          "labelled test sets (protocol.csv; one drifted copy of the same test split per phase).",
          "- 'Offline frozen on phase 1' ~ the `static` policy, but its bank is a coreset of the "
          "first WARMUP images only, not all of phase 1.",
          "- NEW RUN: offline frozen on ALL phase-1 data; offline rebuilt on the whole stream.",
          "- **Pre-shift check FAILS** (preshift_audit.csv): PPO trains on the full stream incl. "
          "post-shift phases, the obs normaliser is fitted on a full episode, and reward fits "
          "with drifted/forget weights use post-shift labels. Needs training restricted to "
          "phase 0 — a code change, not a rerun.",
          (f"- PRESHIFT=1 runs on disk: {len(pre_runs)} (_pre tags; paired per phase in "
           "preshift_pairs.csv / [H])." if pre_runs else ""), ""]

    n_perm = sum(1 for c in calib if c["B_permutations"] != "")
    L += ["## 5. Calibration", "",
          f"- COVERED: {len(calib)} fits in calibration.csv ({n_perm} with a permutation null).",
          "- Before/after retargeting: compare rows of one class with different `target` values.",
          "- Eq. 11: the code has 7 weighted terms + 1 threshold (churn_budget) = 8 numbers; "
          "reconcile the paper's 'six terms' against hyperparams.json.",
          "- **Labels came from the TEST split** (traces score the cached test/stage_* sets, the "
          "same images reported). NEW RUN: calibrate on a disjoint split, then refit, retrain "
          "and re-benchmark everything downstream.", ""]

    imnet = sorted({r["run"] for r in runs if r["dataset"] != "mvtec"
                    and r["backbone"] in ("pvtv2_b2", "wideresnet50")})
    L += ["## 6. Backbone check", ""]
    for row in overlap:
        if row["comparison"].endswith("polyp_pvt_train"):
            L.append(f"- {row['comparison']}: {row['split_a']} -> {row['split_b']}: "
                     f"{row['any_match']}/{row['n_a']} matched (md5 {row['md5_dupes']}, "
                     f"name {row['filename_dupes']}, dHash {row['dhash_candidates']})")
    if not any(r["comparison"].endswith("polyp_pvt_train") for r in overlap):
        L.append("- Polyp-PVT train overlap NOT computed: pass --polyp_pvt_train_dir. Expect it "
                 "to be high: Kvasir's anomalies are Kvasir-SEG (prepare_dataset.py) and "
                 "Polyp-PVT trains on 900 of the 1000 Kvasir-SEG images.")
    L += [f"- ImageNet-only backbone runs on endoscopy: {imnet or 'NONE -> NEW RUN (BACKBONE=pvtv2_b2)'}", ""]

    L += ["## 7. Deployment cost", "",
          "- COVERED (analytic): bank / mirror / policy / window / probe-set sizes and backbone "
          "params in deployment_memory.csv.",
          (f"- COVERED (measured): deployment.csv in {len({d['run'] for d in deploy_meas})} run(s) — "
           "feature / NN search / maintenance ms and peak GPU MiB in deployment_measured.csv / [G]."
           if deploy_meas else
           "- NEW RUN: measured peak GPU memory and per-image latency split into feature "
           "extraction / NN search / maintenance step (logs only hold training-time cost)."), ""]

    L += ["## 8. Protocol", "",
          "- COVERED: counts, anomaly types, warmup and eval point in protocol.csv; method "
          "details in hyperparams.json; Kvasir/HyperKvasir overlap in overlap_kvasir_hyperkvasir*.csv.",
          "- Mask origin (from prep scripts): Kvasir = Kvasir-SEG masks; HyperKvasir = "
          "segmented-images/masks. Normals: Kvasir-v2 normal-{cecum,pylorus,z-line}; HyperKvasir "
          "anatomical landmarks (6 classes).",
          "- Learning rate is not logged anywhere — confirm from the submit command.", ""]
    return "\n".join(l for l in L if l is not None)


# ---------------------------------------------------------------------------
# Plain-text report: the same numbers as the CSVs, as box tables.
# ---------------------------------------------------------------------------
def box(title, header, rows):
    rows = [["" if v is None else str(v) for v in r] for r in rows]
    if not rows:
        return f"{title}\n  (no data)\n"
    w = [max(len(str(h)), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    line = lambda l, m, r: l + m.join("─" * (x + 2) for x in w) + r  # noqa: E731
    fmt = lambda cells: "│" + "│".join(f" {c:<{x}} " for c, x in zip(cells, w)) + "│"  # noqa: E731
    out = [title, line("┌", "┬", "┐"), fmt(header), line("├", "┼", "┤")]
    out += [fmt(r) for r in rows] + [line("└", "┴", "┘")]
    return "\n".join(out) + "\n"


def pm(mean, std):
    if not isinstance(mean, float) or not np.isfinite(mean):
        return "–"
    return f"{mean:.3f}±{std:.3f}" if isinstance(std, float) and np.isfinite(std) else f"{mean:.3f}"


def _budget_key(b):
    """'2.5% match' < '10% match' < 'M=400' < 'M=2000' < 'M=?'."""
    m = re.match(r"([\d.]+)% match", b or "")
    if m:
        return (0, float(m.group(1)))
    m = re.match(r"M=(\d+)", b or "")
    return (1, int(m.group(1))) if m else (2, 0)


def render_report(ds_mean, sweep, stock, per_seed, drift_rows, preshift, calib,
                  overlap, deploy, protocol, gaps, deploy_meas=(), full=True):
    """full=False: per-class tables collapse to per-(dataset, budget) summaries
    and per-policy tables pivot to one row per budget, so the whole report fits
    in one paste; full=True keeps every row."""
    S = ["=" * 78, " REVIEWER CHECKLIST — HARVESTED FROM EXISTING RUNS"
         + ("" if full else "  (short: full per-class tables in report.txt)"), "=" * 78, ""]
    S += ["-" * 78, " ι-REWARD PROTOCOL (tags _<budget>_iota[_k<κ>][_pre])", "-" * 78]
    S += render_sections(new_sections(ds_mean, per_seed, list(deploy_meas), full))
    S += ["-" * 78, " ALL RUNS (legacy proxy-reward runs and ι runs alike)", "-" * 78]

    if full:
        # 1. memory sweep: cross-class mean ± std at stage 0, best image AUROC first per budget
        S.append("[1] MEMORY SWEEP — stage-0, mean ± std over (class, seed)")
        rows = sorted(ds_mean, key=lambda d: (d["dataset"], d["backbone"], _budget_key(d["budget"]), d["variant"],
                                              -(d["image_auroc_mean"] if isinstance(d["image_auroc_mean"], float)
                                                and np.isfinite(d["image_auroc_mean"]) else -1)))
        S.append(box("", ["dataset", "backbone", "budget", "variant", "policy", "image AUROC", "pixel AUROC",
                          "PRO", "writes", "n"],
                     [[d["dataset"], d["backbone"], d["budget"], d["variant"], d["policy"],
                       pm(d["image_auroc_mean"], d["image_auroc_std"]),
                       pm(d["pixel_auroc_mean"], d["pixel_auroc_std"]),
                       pm(d["pro_mean"], d["pro_std"]), d["writes_mean"], d["n_class_seed_pairs"]]
                      for d in rows]))
    else:
        # one row per (dataset, budget, variant), one column per policy: the
        # long form is ~60k chars once the ι budget x κ sweep lands
        S.append("[1] MEMORY SWEEP — stage-0 image AUROC (writes) per streaming policy, cross-class "
                 "mean, uncapped non-pre ι runs; ± std, pixel, PRO, κ/_pre variants in report.txt")
        ds_iota = [d for d in ds_mean if _main_iota(d)]  # κ sweep: [B]; _pre: [H]
        n_legacy = len({x["run"] for x in per_seed if x["arm"] == "legacy"})
        if n_legacy:
            S.append(f"    {n_legacy} legacy proxy-reward run dirs omitted from short [1]-[3]; in report.txt.")
        pols = _policy_order({d["policy"] for d in ds_iota} - set(NOT_STREAMING))  # those: [E]
        cell = lambda d: _f(d["image_auroc_mean"]) + (f" ({kw(d['writes_mean'])})"  # noqa: E731
                                                      if d["writes_mean"] != "" else "")
        S.append(box("", ["dataset", "backbone", "budget", "variant", "n"] + [_abbr(p) for p in pols],
                     [[*k, max(d["n_class_seed_pairs"] for d in v.values()),
                       *(cell(v[p]) if p in v else "" for p in pols)] for k, v in _by_group(ds_iota).items()]))
    if stock:
        g = defaultdict(list)
        for x in stock:
            g[(os.path.relpath(x["source"], os.path.dirname(os.path.dirname(os.path.dirname(x["source"])))),
               x["coreset_pct"])].append(x)
        S.append(box("Offline stock PatchCore (full train set, all classes in file)",
                     ["results.csv", "coreset %", "classes", "image AUROC", "pixel AUROC", "PRO"],
                     [[c, pct, len(v), pm(*mean_std([x["image_auroc"] for x in v])[:2]),
                       pm(*mean_std([x["pixel_auroc"] for x in v])[:2]),
                       pm(*mean_std([x["pro"] for x in v])[:2])]
                      for (c, pct), v in sorted(g.items())]))

    # 2. seed variance: every seed's stage-0 image/pixel AUROC per class & policy
    if not full:
        S.append("[2] SEED VARIANCE — mean over classes of the per-class std of stage-0 image AUROC "
                 "across seeds (eval x train), per policy; uncapped non-pre ι runs")
        g = defaultdict(list)
        for x in per_seed:
            if x["stage"] == "0" and _main_iota(x):
                g[(x["dataset"], x["backbone"], x["budget"], x["variant"], x["policy"], x["class"])].append(
                    x["image_auroc"])
        h = defaultdict(lambda: defaultdict(list))
        for (*k, pol, c), v in g.items():
            h[tuple(k)][pol].append((mean_std(v)[1], c, len(v)))
        pols = _policy_order({k[4] for k in g})
        rows = []
        for k, v in sorted(h.items(), key=lambda kv: _gkey(kv[0])):
            worst = max(((sd, _abbr(p), c) for p, l in v.items() for sd, c, _ in l if np.isfinite(sd)),
                        default=(float("nan"), "–", "–"))
            rows.append([*k, max(n for l in v.values() for _, _, n in l),
                         *(_f(float(np.nanmean([sd for sd, _, _ in v[p]]))) if p in v else "" for p in pols),
                         f"{worst[1]}:{worst[2]} {_f(worst[0])}"])
        S.append(box("", ["dataset", "backbone", "budget", "variant", "seeds"] + [_abbr(p) for p in pols]
                     + ["worst policy:class"], rows))
    else:
        S += _seed_table(per_seed)

    return _render_rest(S, per_seed, drift_rows, preshift, calib, overlap, deploy,
                        protocol, gaps, full, deploy_meas)


def _seed_table(per_seed):
    S = ["[2] SEED VARIANCE — stage-0 per class, per seed (image AUROC / pixel AUROC)"]
    g = defaultdict(list)
    for x in per_seed:
        if x["stage"] == "0":
            g[(x["dataset"], x["budget"], x["variant"], x["class"], x["capacity"], x["policy"])].append(x)
    S.append(box("", ["dataset", "budget", "variant", "class", "M", "policy", "image per seed", "image", "pixel per seed", "pixel"],
                 [[ds, b, var, c, M, p,
                   " ".join(f"{x['image_auroc']:.3f}" for x in sorted(v, key=lambda x: x["seed"])),
                   pm(*mean_std([x["image_auroc"] for x in v])[:2]),
                   " ".join(f"{x['pixel_auroc']:.3f}" for x in sorted(v, key=lambda x: x["seed"])),
                   pm(*mean_std([x["pixel_auroc"] for x in v])[:2])]
                  for (ds, b, var, c, M, p), v in sorted(g.items(), key=lambda kv: (kv[0][0], _budget_key(kv[0][1]), kv[0][2], kv[0][3], kv[0][5]))]))
    return S


def _render_rest(S, per_seed, drift_rows, preshift, calib, overlap, deploy, protocol, gaps, full,
                 deploy_meas=()):
    # 3. full stream: image AUROC at each quarter, cross-class mean
    S.append("[3] FULL STREAM — image AUROC at each point of the stream (cross-class mean ± std)"
             + ("" if full else "; short: uncapped ι runs, budgets pooled, phase columns only "
                                "(s0@X in [D])"))
    S.append("    25/50/75/100% = bank at end of that quarter, scored on THAT phase's test set;")
    S.append("    s0@X = bank at X% of the stream scored on the STAGE-0 test set "
             "(s0@100% = forgetting; s0@50/75% need STAGE0_EVERY=1 runs).")
    g = defaultdict(lambda: defaultdict(list))
    for x in per_seed:
        if not full and not _main_iota(x):
            continue
        pct = f"{int(round(100 * x['stream_frac_evaluated_at']))}%"
        col = ("s0@100%" if x["stage"] == FORGET else
               f"s0@{pct}" if x["stage"].startswith("stage0_at_") else
               "s0@25%" if x["stage"] == "0" else pct)
        key = ((x["dataset"], x["backbone"], x["budget"], x["variant"], x["policy"]) if full
               else (x["dataset"], x["backbone"], x["policy"]))
        if x["stage"] == "0":
            g[key]["25%"].append(x["image_auroc"])
        g[key][col].append(x["image_auroc"])
    if full:
        cols = ["25%", "50%", "75%", "100%", "s0@25%", "s0@50%", "s0@75%", "s0@100%"]
        S.append(box("", ["dataset", "backbone", "budget", "variant", "policy"] + cols,
                     [[*k, *(pm(*mean_std(v[c])[:2]) for c in cols)]
                      for k, v in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1], _budget_key(kv[0][2]), kv[0][3], kv[0][4]))]))
    else:
        cols, order = ["25%", "50%", "75%", "100%"], list(POLICY_ABBR)
        S.append(box("", ["dataset", "backbone", "policy"] + cols,
                     [[*k, *(_f(mean_std(v[c])[0]) for c in cols)]
                      for k, v in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1], order.index(kv[0][2])
                                                                    if kv[0][2] in order else len(order)))]))

    # 4. drift
    S.append("[4] DRIFT — phase definitions (staged_abrupt_4) and pre-shift audit")
    S.append(box("", ["phase", "when", "wb_gain", "hue°", "gamma", "blur σ", "specular", "vignette", "jitter"],
                 [[d["phase"], d["when"].split(" of")[0], d["wb_gain"], d["hue_shift_deg"], d["gamma"],
                   d["blur_sigma"], d["specular"], d["vignette"], d["jitter"]]
                  for d in drift_rows if d["schedule"] == "staged_abrupt_4"]))
    n_bad = sum(1 for r in preshift if r["verdict"].startswith("NOT"))
    S.append(f"    Pre-shift-only: {n_bad}/{len(preshift)} runs FAIL "
             "(PPO trains on all phases; obs-norm fitted on a full episode).")
    S.append(f"    Runs whose reward-fit target uses post-shift labels: "
             f"{sum(1 for r in preshift if r['fit_target_uses_postshift_labels'] is True)}/{len(preshift)}\n")

    # 5. calibration (proxy-reward fits only; the ι reward has no fitted weights)
    if not full:
        ps = [c for c in calib if not isinstance(c["p_value"], str)]
        S.append(f"[5] CALIBRATION — {len(calib)} legacy proxy-reward fits, {len(ps)} with a permutation "
                 f"null, {sum(c['p_value'] <= 0.05 for c in ps)} at p≤.05; tables in report.txt / "
                 "calibration.csv (the ι reward fits nothing)")
    wk = ("alpha", "beta", "gamma", "churn_coef", "churn_budget", "c90_coef", "probe_coef", "q_coef")
    if full:
        S.append("[5] CALIBRATION — reward fit per run (labels from the TEST split)")
        S.append(box("", ["run", "target", "J", "ρ*", "p", "B", "null p95", "p≤.05",
                          "w* (α β γ churn budget c90 probe q)"],
                     [[c["run"], c["target"], c["J_n_traces"], c["rho_star"],
                       c["p_value"] if not isinstance(c["p_value"], str) else "–", c["B_permutations"],
                       c["null_p95"], c["pass_p<=0.05"], " ".join(str(c.get(f"w_{k}", "")) for k in wk)]
                      for c in sorted(calib, key=lambda c: c["run"])]))
    S.append("    J, reference trajectories and w* per run: calibration.csv / report.txt.\n")

    # 6 / 8. overlap
    S.append("[6/8] DUPLICATE IMAGES — A image -> its match in B (dHash = near-duplicate candidate)")
    S.append(box("", ["comparison", "A split", "n A", "B split", "md5", "filename", "dHash", "any"],
                 [[o["comparison"], o["split_a"], o["n_a"], o["split_b"], o["md5_dupes"],
                   o["filename_dupes"], o["dhash_candidates"], o["any_match"]] for o in overlap]))

    # 7. deployment
    if not full:
        seen, keep = set(), []
        for d in sorted(deploy, key=lambda d: -int(d["M"])):
            key = (d["backbone"], d["D"])
            if key not in seen:  # largest M per backbone = worst case
                seen.add(key)
                keep.append(d)
        deploy = keep
    S.append("[7] DEPLOYMENT MEMORY — analytic fp32 MiB"
             + (" (measured latency + peak memory: [G])" if deploy_meas else
                " (latency not measured by any existing run)")
             + ("" if full else "; largest M per backbone shown"))
    S.append(box("", ["run", "M", "D", "backbone params", "backbone", "bank", "policy", "window",
                      "probe Z", "total", "train ms/step"],
                 [[d["run"], d["M"], d["D"], d["backbone_params"], d["backbone_MiB_fp32"], d["bank_MiB"],
                   d["policy_MiB"], d["window_MiB"], d["probe_Z_MiB"],
                   round(sum(float(d[k]) for k in ("backbone_MiB_fp32", "bank_MiB", "policy_MiB",
                                                   "window_MiB", "probe_Z_MiB") if d[k] != ""), 3),
                   d.get("train_ms_per_env_step", "")] for d in deploy]))

    # 8. protocol
    if not full:
        seen, keep = set(), []
        for p_ in protocol:
            key = (p_["class"], p_["backbone"], p_["warmup_images"])
            if key not in seen:
                seen.add(key)
                keep.append(p_)
        iota_runs = {x["run"] for x in per_seed if x["arm"] == "iota"}
        protocol = [p_ for p_ in keep if p_["run"] in iota_runs] or keep
    S.append("[8] PROTOCOL — counts, warmup and stage-0 eval point"
             + ("" if full else " (one row per class/warmup; ι runs when any exist)"))
    S.append(box("", ["run", "train", "per phase", "warmup", "eval after img", "policy steps",
                      "test normal", "test abnormal", "masks", "anomaly types"],
                 [[p["run"], p["n_train_stream"], p["images_per_phase"], p["warmup_images"],
                   p["stage0_eval_after_image"], p["stage0_policy_steps"], p["test_normal"],
                   p["test_abnormal"], p["test_nonempty_masks"], p["anomaly_types"]] for p in protocol]))
    S.append("    Method details (E, MLP, decoding, eviction, τ κ |Z| window W B, lr, episodes): hyperparams.json\n")

    # short: only the ι-protocol coverage; the legacy checklist lives in GAPS.md
    S += ["=" * 78, gaps if full else gaps.split(LEGACY_GAPS_MARK)[0]
          + "Sections 1-8 (original reviewer checklist, legacy runs): GAPS.md / report.txt.\n"]
    return "\n".join(S)


# ---------------------------------------------------------------------------
# ι-reward protocol sections [A]-[H]. Each builder returns
# (csv name, title, one-line explanations, header, rows); full=False is the
# compact rendering for report_short.txt, full=True also goes to <csv>.csv.
# ---------------------------------------------------------------------------
POLICY_ABBR = {"static": "static", "fifo": "fifo", "reservoir": "resv", "random": "rand",
               "streaming_greedy_coreset": "sgc", "periodic_coreset": "periodic",
               "kcenter": "kctr", "myopic_iota": "myopic", "ppo": "ppo",
               "offline_stage0": "off_s0", "offline_all": "off_all",
               "full_stage0": "full_s0", "full_all": "full_all"}
NOT_STREAMING = ("offline_stage0", "offline_all", "full_stage0", "full_all")


def _abbr(p):
    return POLICY_ABBR.get(p, p)


def _policy_order(pols):
    order = list(POLICY_ABBR)
    return sorted(pols, key=lambda p: (order.index(p) if p in order else len(order), p))


def _num(x):
    return x if isinstance(x, float) and np.isfinite(x) else float("-inf")


def _f(x, fmt=".3f"):
    return format(x, fmt) if isinstance(x, float) and np.isfinite(x) else "–"


def kw(w):
    """Compact write count: 950, 2.4k, 1.31M."""
    if w in ("", None):
        return "–"
    w = float(w)
    return f"{w:.0f}" if w < 1e3 else f"{w / 1e3:.1f}k" if w < 1e6 else f"{w / 1e6:.2f}M"


def _spearman(a, b):
    from scipy import stats

    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(stats.spearmanr(a, b).correlation)


def _gkey(k):
    """Sort (dataset, backbone, budget, ...) keys with budgets in size order."""
    return (k[0], k[1], _budget_key(k[2]), *(str(x) for x in k[3:]))


def _by_group(ds_mean, keep=lambda d: True):
    """(dataset, backbone, budget, variant) -> policy -> ds_mean row."""
    by = defaultdict(dict)
    for d in ds_mean:
        if keep(d):
            by[(d["dataset"], d["backbone"], d["budget"], d["variant"])][d["policy"]] = d
    return dict(sorted(by.items(), key=lambda kv: _gkey(kv[0])))


def _main_iota(d):
    """The headline ι runs: no admit cap, not pre-shift."""
    return d["arm"] == "iota" and d["kappa"] == INF and not d["preshift"]


def sec_iota_validation(per_seed, full):
    rows = [x for x in per_seed if x["arm"] == "iota" and x["stage"] == "0"
            and np.isfinite(x["iota"]) and np.isfinite(x["image_auroc"])]
    groups = defaultdict(list)
    for x in rows:
        groups[(x["dataset"], "all")].append(x)
        groups[("ALL", "all")].append(x)
        if full:
            groups[(x["dataset"], x["budget"])].append(x)
    out = []
    for (ds, b), g in sorted(groups.items(), key=lambda kv: (kv[0][0] == "ALL", kv[0][0],
                                                              kv[0][1] != "all", _budget_key(kv[0][1]))):
        by_cls = defaultdict(list)
        for x in g:
            by_cls[x["class"]].append(x)
        rho_c = [r for r in (_spearman([-x["iota"] for x in v], [x["image_auroc"] for x in v])
                             for v in by_cls.values()) if np.isfinite(r)]
        out.append([ds, b, len(g), len(by_cls), len({x["policy"] for x in g}),
                    _f(_spearman([-x["iota"] for x in g], [x["image_auroc"] for x in g])),
                    _f(float(np.median(rho_c))) if rho_c else "–",
                    f"{sum(r > 0 for r in rho_c)}/{len(rho_c)}"])
    return ("iota_validation", "[A] ι VALIDATION — does a lower ι go with a higher stage-0 image AUROC?",
            ["Spearman ρ(-ι at the stage-0 boundary, stage-0 image AUROC) over every (class, policy, "
             "seed) row of ι-arm runs; pure correlation, nothing fitted.",
             "Pooled ρ mixes class difficulty; within-class ρ is the clean test "
             "(classes ρ>0 = classes where ι ranks the banks the right way)."],
            ["dataset", "budget", "rows", "classes", "policies", "pooled ρ", "median class ρ",
             "classes ρ>0"], out)


def sec_kappa(ds_mean, full):
    by = defaultdict(dict)  # (dataset, backbone, budget) -> κ -> policy -> row
    for d in ds_mean:
        if d["arm"] == "iota" and not d["preshift"]:
            by[(d["dataset"], d["backbone"], d["budget"])].setdefault(d["kappa"], {})[d["policy"]] = d
    out = []
    for k, kap in sorted(by.items(), key=lambda kv: _gkey(kv[0])):
        if len(kap) < 2:  # no sweep at this budget
            continue
        for kv_ in sorted(kap, key=_kappa_key):
            if (d := kap[kv_].get("ppo")):
                out.append([*k, kv_, "ppo", pm(d["image_auroc_mean"], d["image_auroc_std"]),
                            kw(d["writes_mean"]), d["n_class_seed_pairs"]])
        ref = kap.get(INF, {})
        base = [p for p in ref if p != "ppo"]
        if not full:  # static + the best streaming baseline only
            best = max((p for p in base if p not in NOT_STREAMING),
                       key=lambda p: _num(ref[p]["image_auroc_mean"]), default=None)
            base = list(dict.fromkeys(p for p in ("static", best) if p in ref))
        for p in _policy_order(base):
            d = ref[p]
            out.append([*k, f"{INF} ref", p, pm(d["image_auroc_mean"], d["image_auroc_std"]),
                        kw(d["writes_mean"]), d["n_class_seed_pairs"]])
    return ("kappa_pareto", "[B] κ PARETO — stage-0 accuracy vs writes as the admit cap κ tightens",
            ["PPO per κ (patches admitted per image, tags _iota_k<κ>); 'ref' rows = baselines of the "
             "uncapped run. A good κ keeps AUROC with far fewer writes."
             + ("" if full else " Short: static + best baseline; all in report.txt.")],
            ["dataset", "backbone", "budget", "κ", "policy", "image AUROC", "writes", "n"], out)


def sec_train_seeds(per_seed, full):
    g = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for x in per_seed:
        if (x["policy"] == "ppo" and x["stage"] == "0" and x["train_seed"] != ""
                and np.isfinite(x["image_auroc"]) and (full or _main_iota(x))):
            g[(x["dataset"], x["backbone"], x["budget"], x["variant"])][x["class"]][x["train_seed"]].append(
                x["image_auroc"])
    out = []
    for k, classes in sorted(g.items(), key=lambda kv: _gkey(kv[0])):
        per = []
        for c, ts in sorted(classes.items()):
            if len(ts) < 2:
                continue
            means = [float(np.mean(ts[t])) for t in sorted(ts)]
            ev = [np.std(v, ddof=1) for v in ts.values() if len(v) > 1]
            per.append((c, len(ts), max(len(v) for v in ts.values()), float(np.std(means, ddof=1)),
                        float(np.mean(ev)) if ev else float("nan"), means))
        if full:
            out += [[*k, c, nt, ne, _f(st), _f(se), " ".join(f"{m:.3f}" for m in ms)]
                    for c, nt, ne, st, se, ms in per]
        elif per:
            st, se = float(np.mean([p[3] for p in per])), float(np.nanmean([p[4] for p in per]))
            worst = max(per, key=lambda p: p[3])
            out.append([*k, len(per), max(p[1] for p in per), max(p[2] for p in per), _f(st), _f(se),
                        _f(st / se, ".2f") if se > 0 else "–", f"{worst[0]} {worst[3]:.3f}"])
    head = (["dataset", "backbone", "budget", "variant", "class", "train seeds", "eval seeds",
             "std train", "std eval", "mean per train seed"] if full else
            ["dataset", "backbone", "budget", "variant", "classes", "train seeds", "eval seeds",
             "std train", "std eval", "train/eval", "worst class"])
    return ("train_seed_variance", "[C] TRAIN-SEED VARIANCE — PPO stage-0 image AUROC",
            ["std train = std over training seeds of the eval-seed mean; std eval = std over eval "
             "seeds with the policy fixed (mean over train seeds). Ratio >1: training is the bigger "
             "noise source." + ("" if full else " Uncapped non-pre runs, means over classes; "
                                                "per class and variant in report.txt.")],
            head, out)


def sec_anytime(per_seed, full):
    auc = defaultdict(lambda: defaultdict(list))
    io = defaultdict(lambda: defaultdict(dict))  # key -> pct -> sample -> ι (one per boundary)
    for x in per_seed:
        if x["arm"] != "iota" or (not full and (not _main_iota(x) or x["policy"] in NOT_STREAMING)):
            continue
        key = ((x["dataset"], x["backbone"], x["budget"], x["variant"], x["policy"]) if full
               else (x["dataset"], x["backbone"], x["policy"]))
        pct = int(round(100 * x["stream_frac_evaluated_at"]))
        if x["test_set"] == "stage_0" and np.isfinite(x["image_auroc"]):
            auc[key][pct].append(x["image_auroc"])
        if np.isfinite(x["iota"]):  # stage s and stage0_at_s rows share one boundary ι
            io[key][pct].setdefault((x["run"], x["seed"], x["train_seed"]), x["iota"])
    pcts = sorted({p for v in list(auc.values()) + list(io.values()) for p in v})
    order = list(POLICY_ABBR)
    keys = sorted(set(auc) | set(io), key=lambda k: (
        (k[0], k[1], _budget_key(k[2]), k[3]) if full else (k[0], k[1]),
        order.index(k[-1]) if k[-1] in order else len(order), k[-1]))
    cell = (lambda v: pm(*mean_std(v)[:2])) if full else (lambda v: _f(mean_std(v)[0]))
    out = [[*k[:-1], k[-1], *(cell(auc[k][p]) for p in pcts),
            *(_f(mean_std(list(io[k][p].values()))[0], ".3g") for p in pcts)] for k in keys]
    head = (["dataset", "backbone", "budget", "variant"] if full else ["dataset", "backbone"]) + ["policy"]
    return ("anytime", "[D] ANYTIME — stage-0 image AUROC and ι at each point of the stream",
            ["s0@X = bank at X% of the stream scored on the stage-0 test set (needs "
             "--stage0_every_boundary for 50/75%); ι@X = mean ι at that boundary. Flat s0 = no forgetting."
             + ("" if full else " Short: uncapped ι runs, budgets pooled, streaming policies; "
                                "per budget in report.txt.")],
            head + [f"s0@{p}%" for p in pcts] + [f"ι@{p}%" for p in pcts], out)


def sec_less_is_more(ds_mean, full):
    out = []
    for k, pols in _by_group(ds_mean).items():
        if not any(p in pols for p in NOT_STREAMING):
            continue
        a = lambda p: pols[p]["image_auroc_mean"] if p in pols else float("nan")  # noqa: E731
        cell = ((lambda p: pm(pols[p]["image_auroc_mean"], pols[p]["image_auroc_std"]) if p in pols else "–")
                if full else (lambda p: _f(a(p))))
        best = max((p for p in pols if p not in NOT_STREAMING), key=lambda p: _num(a(p)), default=None)
        delta = a(best) - a("full_stage0") if best else float("nan")
        out.append([*k, cell("static"), f"{_abbr(best)} {_f(a(best))}" if best else "–",
                    *(cell(p) for p in NOT_STREAMING), _f(delta, "+.3f")])
    return ("less_is_more", "[E] LESS IS MORE? — stage-0 image AUROC: small streaming bank vs keeping everything",
            ["full_* = every stream patch kept (no coreset; stage-0 images / whole stream); offline_* = "
             "greedy coreset at M. Δ = best streaming − full_stage0: Δ ≥ 0 means a small bank loses nothing."],
            ["dataset", "backbone", "budget", "variant", "static", "best streaming",
             *(_abbr(p) for p in NOT_STREAMING), "Δ best−full_s0"], out)


def sec_rl_vs_myopic(ds_mean, full):
    out = []
    keep = (lambda d: d["arm"] == "iota") if full else _main_iota
    for k, pols in _by_group(ds_mean, keep).items():
        if not any(p in pols for p in ("myopic_iota", "kcenter", "ppo")):
            continue
        row = [*k, _f(pols["static"]["image_auroc_mean"]) if "static" in pols else "–"]
        for p in ("ppo", "myopic_iota", "kcenter"):
            d = pols.get(p)
            row += [pm(d["image_auroc_mean"], d["image_auroc_std"]) if d else "–",
                    kw(d["writes_mean"]) if d else "–"]
        out.append(row)
    return ("rl_vs_myopic", "[F] RL vs MYOPIC vs K-CENTER — stage-0 image AUROC and writes per budget",
            ["Does the learned policy beat the label-free one-step ι lookahead (myopic) and streaming "
             "k-center at the same budget? static = no-maintenance floor."
             + ("" if full else " Short: uncapped, non-pre runs.")],
            ["dataset", "backbone", "budget", "variant", "static", "ppo", "ppo writes", "myopic",
             "myopic writes", "kcenter", "kctr writes"], out)


def harvest_deployment_measured(runs):
    return [{"run": r["run"], "dataset": r["dataset"], "budget": r["budget"], "variant": r["variant"],
             **{k: row.get(k, "") for k in ("gpu", "M", "D", "policy", "feature_ms", "nn_search_ms",
                                            "maintenance_ms", "total_ms", "peak_gpu_MiB")}}
            for r in runs for row in r["deploy_measured"]]


def sec_deployment(deploy_meas, full):
    if full:
        out = [[d["run"], d["gpu"], d["M"], d["D"], d["policy"], d["feature_ms"], d["nn_search_ms"],
                d["maintenance_ms"], d["total_ms"], d["peak_gpu_MiB"]] for d in deploy_meas]
        head = ["run", "gpu", "M", "D", "policy", "feature ms", "NN search ms", "maintenance ms",
                "total ms", "peak GPU MiB"]
    else:
        g = defaultdict(list)
        for d in deploy_meas:
            g[d["run"]].append(d)
        out = [[run, v[0]["M"], v[0]["feature_ms"], v[0]["nn_search_ms"], v[0]["peak_gpu_MiB"],
                " ".join(f"{_abbr(d['policy'])} {d['maintenance_ms']}" for d in v)] for run, v in g.items()]
        head = ["run", "M", "feature ms", "NN ms", "peak MiB", "maintenance ms/step per policy"]
    return ("deployment_measured", "[G] DEPLOYMENT — measured per-image latency and peak GPU memory (deployment.csv)",
            ["Per image: backbone features + k-NN scoring against the bank + one maintenance step of the "
             "policy; peak = max GPU memory allocated over the whole measurement."], head, out)


def sec_preshift(per_seed, full):
    present = {x["run"] for x in per_seed}
    meta, vals = {}, defaultdict(list)
    for x in per_seed:
        meta.setdefault(x["run"], x)
        col = "forget" if x["stage"] == FORGET else x["stage"] if x["stage"].isdigit() else None
        if col is not None:
            vals[(x["run"], x["policy"], col)].append(x["image_auroc"])
    cols = sorted({c for _, _, c in vals if c != "forget"}, key=int) + ["forget"]
    agg = defaultdict(lambda: defaultdict(lambda: ([], [])))  # key -> col -> (pre, twin) class means
    for run, m in sorted(meta.items()):
        if not m["preshift"]:
            continue
        base = run[: -len("_pre")]
        # submit_reviewer_runs.sh pairs legacy _M2000_pre with _M2000_rv
        twin = next((t for t in ([base + "_rv", base] if m["arm"] == "legacy" else [base])
                     if t in present), None)
        if twin is None:
            continue
        for pol in {p for r_, p, _ in vals if r_ == run}:
            if not full and pol != "ppo":
                continue
            key = (m["dataset"], m["backbone"], m["budget"], m["arm"], m["kappa"], pol)
            for c in cols:
                a, b = vals.get((run, pol, c)), vals.get((twin, pol, c))
                if a and b:
                    agg[key][c][0].append(float(np.nanmean(a)))
                    agg[key][c][1].append(float(np.nanmean(b)))
    order = list(POLICY_ABBR)
    out = [[*k, len(v[cols[0]][0]) if cols[0] in v else 0,
            *(f"{_f(float(np.mean(v[c][0])))}/{_f(float(np.mean(v[c][1])))}" if c in v else "–" for c in cols)]
           for k, v in sorted(agg.items(), key=lambda kv: (*_gkey(kv[0][:-1]), order.index(kv[0][-1])
                                                           if kv[0][-1] in order else len(order)))]
    return ("preshift_pairs", "[H] PRE-SHIFT DRIFT — per-phase image AUROC, _pre run / its full-stream twin",
            ["_pre = PPO trained on the pre-shift stream only, benchmarked on all phases (each phase on "
             "its own test set; forget = stage-0 test with the final bank). Class-paired means."
             + ("" if full else " Short: PPO only.")],
            ["dataset", "backbone", "budget", "arm", "κ", "policy", "classes",
             *(f"phase {c}" if c != "forget" else "forget" for c in cols)], out)


def new_sections(ds_mean, per_seed, deploy_meas, full):
    return [sec_iota_validation(per_seed, full), sec_kappa(ds_mean, full),
            sec_train_seeds(per_seed, full), sec_anytime(per_seed, full),
            sec_less_is_more(ds_mean, full), sec_rl_vs_myopic(ds_mean, full),
            sec_deployment(deploy_meas, full), sec_preshift(per_seed, full)]


def render_sections(secs):
    S = []
    for _, title, notes, header, rows in secs:
        S += [title] + [f"    {n}" for n in notes] + [box("", header, rows)]
    return S


def main():
    import argparse

    pkg_default = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pkg_dir", default=pkg_default)
    ap.add_argument("--log_dir", action="append", default=None,
                    help="repeatable; default: every existing one of <pkg>/../../logs, "
                         "<pkg>/../logs, <pkg>/logs")
    ap.add_argument("--dataset_root", default=os.environ.get(
        "DATASET_ROOT", "/home/user1/aniket/Patchcore/dataset"))
    ap.add_argument("--polyp_pvt_train_dir", default=None,
                    help="Polyp-PVT TrainDataset/image dir for the test-set overlap count")
    ap.add_argument("--skip_overlap", action="store_true")
    ap.add_argument("--backfill_permute", type=int, default=200,
                    help="permutation-null shuffles for fits recorded without one "
                         "(from their reward_traces.pkl; 0 = off)")
    ap.add_argument("--out", default=None, help="default: <pkg>/results/harvest")
    a = ap.parse_args()

    pkg = a.pkg_dir
    out = a.out or os.path.join(pkg, "results", "harvest")
    os.makedirs(out, exist_ok=True)
    # run_streaming.sh logs to ../logs relative to the sbatch SUBMIT dir (the
    # repo root for submit_all_streaming.sh), i.e. one level above the repo
    log_dirs = a.log_dir or [d for d in (os.path.join(pkg, "..", "..", "logs"),
                                         os.path.join(pkg, "..", "logs"),
                                         os.path.join(pkg, "logs")) if os.path.isdir(d)]
    print(f"[logs] parsing {[os.path.realpath(d) for d in log_dirs] or 'NO log dir found'}")
    log_facts, n_logs = parse_logs(log_dirs)
    print(f"[logs] {n_logs} streaming_output_*.log files")
    runs = discover_runs(os.path.join(pkg, "results", "streaming"),
                         os.path.join(pkg, "cache"), log_facts)
    print(f"[runs] {len(runs)} result dirs")
    if not runs:
        print("no result dirs found — run from patchcore-inspection/ or pass --pkg_dir")

    write_csv(os.path.join(out, "runs.csv"), [{
        "run": r["run"], "dataset": r["dataset"], "class": r["class"], "backbone": r["backbone"],
        "drift": r["drift"], "suffix": r["suffix"], "budget": r["budget"], "variant": r["variant"],
        "arm": r["arm"], "kappa": r["kappa"], "preshift": r["preshift"], "capacity": r.get("capacity"),
        "warmup": r.get("warmup"), "stage0_only": r.get("stage0_only"), "job": r.get("job"),
        "gpu": r["log"].get("gpu"), "has_results": os.path.exists(os.path.join(r["dir"], "results.csv")),
        "has_fit": bool(r["fit"]), "checkpoint": r["ckpt"] or "",
        "has_deployment_csv": bool(r["deploy_measured"]),
        "cache_found": bool(r["cache"])} for r in runs])

    per_seed = harvest_per_seed(runs)
    write_csv(os.path.join(out, "per_seed.csv"), per_seed)
    # variant in the key: an ι run and a legacy run at the same M are different
    # methods, not extra seeds of one; train seeds pool into the per-class mean
    sweep = summarize(per_seed, ["dataset", "backbone", "capacity", "budget", "variant", "arm",
                                 "kappa", "preshift", "class", "policy"],
                      lambda x: x["stage"] == "0")
    write_csv(os.path.join(out, "memory_sweep.csv"), sweep)
    ds_mean = pooled(per_seed, ["dataset", "backbone", "budget", "variant", "arm", "kappa",
                                "preshift", "policy"])
    write_csv(os.path.join(out, "memory_sweep_dataset_mean.csv"), ds_mean)
    stock = harvest_stock(pkg)
    write_csv(os.path.join(out, "stock_patchcore.csv"), stock)
    write_csv(os.path.join(out, "seed_variance.csv"), sorted(
        ({k: x[k] for k in ("dataset", "class", "backbone", "capacity", "variant", "run", "policy",
                            "seed", "train_seed", "image_auroc", "pixel_auroc", "pro", "iota",
                            "writes_admit")}
         for x in per_seed if x["stage"] == "0"),
        key=lambda d: (d["run"], d["policy"], str(d["train_seed"]), d["seed"])))
    write_csv(os.path.join(out, "fullstream.csv"), summarize(
        per_seed, ["dataset", "class", "backbone", "budget", "variant", "capacity", "policy",
                   "stream_frac_evaluated_at", "test_set"], lambda x: True))
    drift_rows, preshift = harvest_drift_protocol(), harvest_preshift(runs)
    write_csv(os.path.join(out, "drift_protocol.csv"), drift_rows)
    write_csv(os.path.join(out, "preshift_audit.csv"), preshift)
    calib = harvest_calibration(runs, a.backfill_permute)
    write_csv(os.path.join(out, "calibration.csv"), calib)
    deploy, protocol = harvest_deployment(runs, pkg), harvest_protocol(runs, a.dataset_root)
    write_csv(os.path.join(out, "deployment_memory.csv"), deploy)
    write_csv(os.path.join(out, "protocol.csv"), protocol)
    deploy_meas = harvest_deployment_measured(runs)
    for name, _, _, header, rows in new_sections(ds_mean, per_seed, deploy_meas, full=True):
        write_csv(os.path.join(out, f"{name}.csv"), [dict(zip(header, r)) for r in rows], header)

    hp = code_hyperparams()
    hp["per_run"] = {r["run"]: {
        "capacity": r.get("capacity"), "warmup": r.get("warmup"), "n_nn": r["log"].get("n_nn"),
        "ppo_iters": (r["log"]["ppo_iters"][-1][1] if r["log"].get("ppo_iters") else None),
        "env_steps": (r["log"]["ppo_iters"][-1][1] * 8 * 256 if r["log"].get("ppo_iters") else None),
        "episodes": (round(r["log"]["ppo_iters"][-1][1] * 8 * 256
                           / max(1, r["cache"]["n_stream"] - r["warmup"]), 1)
                     if r["log"].get("ppo_iters") and r["cache"] and r.get("warmup") is not None
                     and r["cache"]["n_stream"] > r["warmup"] else None),
        "best_iter_restored": r["log"].get("ppo_best_iter"),
        "best_window": r["log"].get("best_window"), "adv_mode": r["log"].get("adv_mode"),
        "clip_mode": r["log"].get("clip_mode"),
        "reward_weights": (r["fit"] or {}).get("recommended"),
    } for r in runs}
    with open(os.path.join(out, "hyperparams.json"), "w") as f:
        json.dump(hp, f, indent=2, default=str)
    print(f"  wrote {os.path.join(out, 'hyperparams.json')}")

    overlap = []
    if not a.skip_overlap:
        kv = os.path.join(a.dataset_root, "kvasir_patchcore")
        hk = os.path.join(a.dataset_root, "hyperkvasir_patchcore")
        idx = {n: image_index(p, n) for n, p in (("kvasir", kv), ("hyperkvasir", hk))
               if os.path.isdir(p)}
        if len(idx) == 2:
            pairs = compare_sets(idx["kvasir"], idx["hyperkvasir"])
            write_csv(os.path.join(out, "overlap_kvasir_hyperkvasir_pairs.csv"), pairs)
            overlap += overlap_summary(pairs, idx["kvasir"][0], "kvasir_vs_hyperkvasir")
        else:
            print(f"  [overlap] kvasir/hyperkvasir dirs not both found under {a.dataset_root}")
        if a.polyp_pvt_train_dir and os.path.isdir(a.polyp_pvt_train_dir):
            ppv = image_index(a.polyp_pvt_train_dir, "polyp_pvt_train")
            for n, ix in idx.items():
                tests = [i for i, p in enumerate(ix[0]) if "/test/" in p]
                sub = ([ix[0][i] for i in tests], [ix[1][i] for i in tests], ix[2][tests])
                pairs = compare_sets(sub, ppv)
                write_csv(os.path.join(out, f"overlap_{n}_test_vs_polyp_pvt_train_pairs.csv"), pairs)
                overlap += overlap_summary(pairs, sub[0], f"{n}_test_vs_polyp_pvt_train")
        write_csv(os.path.join(out, "overlap_summary.csv"), overlap)

    gaps = gaps_report(runs, per_seed, calib, overlap, stock, n_logs, deploy_meas)
    with open(os.path.join(out, "GAPS.md"), "w") as f:
        f.write(gaps)
    args = (ds_mean, sweep, stock, per_seed, drift_rows, preshift, calib,
            overlap, deploy, protocol, gaps, deploy_meas)
    with open(os.path.join(out, "report.txt"), "w") as f:
        f.write(render_report(*args, full=True))
    short = render_report(*args, full=False)
    with open(os.path.join(out, "report_short.txt"), "w") as f:
        f.write(short)
    print("\n" + short)
    print(f"\nshort report -> {os.path.join(out, 'report_short.txt')} ({len(short)} chars)")
    print(f"full report  -> {os.path.join(out, 'report.txt')}   (CSVs alongside)")


if __name__ == "__main__":
    main()
