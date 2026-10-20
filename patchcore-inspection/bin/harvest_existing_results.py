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
        r["capacity"] = r.get("capacity") or lf.get("capacity_log")
        r["log"] = lf
        r["job"] = lf.get("bench_job") or lf.get("job")
        mm = re.match(r"_m([\d.]+)(.*)$", r["suffix"])
        r["budget"] = f"{mm.group(1)}% match" if mm else (
            f"M={r['capacity']}" if r.get("capacity") else "M=?")
        r["variant"] = (mm.group(2) if mm else r["suffix"]) or "(untagged)"
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
            seed = int(row["seed"])
            if row["stage"] == FORGET:
                frac, test_set = 1.0, "stage_0"
            else:
                s = int(row["stage"])
                frac, test_set = (s + 1) / n_stages, f"stage_{s}"
            b = bench.get((row["policy"], seed), {})
            rows.append({
                "dataset": r["dataset"], "class": r["class"], "backbone": r["backbone"],
                "run": r["run"], "budget": r["budget"], "variant": r["variant"],
                "capacity": r.get("capacity"), "warmup": r.get("warmup"),
                "policy": row["policy"], "seed": seed, "stage": row["stage"],
                "stream_frac_evaluated_at": frac, "test_set": test_set,
                "image_auroc": fnum(row["image_auroc"]), "pixel_auroc": fnum(row["pixel_auroc"]),
                "pro": fnum(row["pro"]), "writes_admit": b.get("admit", ""),
                "evict": b.get("evict", ""), "job": r.get("job"),
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


def harvest_calibration(runs):
    rows = []
    for r in runs:
        fit = r["fit"]
        if not fit:
            continue
        pn = fit.get("permutation_null") or {}
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
def gaps_report(runs, per_seed, calib, overlap, stock, n_logs):
    L = ["# Reviewer checklist — what existing runs cover", "",
         f"Scanned {len(runs)} result dirs and {n_logs} SLURM logs. "
         "COVERED = harvested into the CSVs here; NEW RUN = no existing artifact has it.", ""]
    by_ds = defaultdict(lambda: defaultdict(set))  # dataset -> M -> policies
    classes_at = defaultdict(lambda: defaultdict(set))
    for x in per_seed:
        if x["stage"] == "0":
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
    L += [f"- Writes: {has_writes}/{len(per_seed)} per-seed rows have admit counts from logs"
          + ("" if has_writes == len(per_seed) else " (rest: log missing/rotated -> rerun step 5)"),
          f"- Offline PatchCore: stock results found for {sorted({s['config'] for s in stock}) or 'none'}"
          " (full train set, NOT stage-0 data, NOT matched M).",
          "- NEW RUN: offline greedy coreset at every exact M (incl. M=2000) — never run.", ""]

    seeds = defaultdict(set)
    for x in per_seed:
        seeds[x["run"]].add(x["seed"])
    ns = sorted({len(v) for v in seeds.values()})
    L += ["## 2. Seed variance", "",
          f"- COVERED: per-class per-seed rows in seed_variance.csv; eval seeds per run: {ns}.",
          "- NEW RUN: 5 seeds (EVAL_SEEDS=0,1,2,3,4) — only step 5 needs rerunning." if ns and max(ns) < 5 else "",
          "- What the eval seed controls (from code): env rng only — probe-set Z draw, warmup "
          "coreset random start, per-step holdout split, window subsample, obs projection. "
          "It does NOT change stream order (fixed by the cache SEED=0), baseline policy "
          "sampling (reservoir/periodic/random use a hard-coded seed=0), or PPO weights "
          "(one policy trained with TRAIN_SEEDS=0, deterministic mean action). So ppo std = "
          "eval-env noise only, not training variance.", ""]

    full = sorted({x["run"] for x in per_seed if x["stage"] == "3"})
    L += ["## 3. Full-stream behaviour", "",
          f"- COVERED for {len(full)} runs: AUROC at 25/50/75/100% — but each on THAT phase's "
          "drifted test set, plus stage-0 test re-scored at 100% (forgetting). See fullstream.csv.",
          "- NEW RUN: stage-0 test at 50% and 75% (never evaluated), and every STAGE0_ONLY run "
          "(only the 25% point exists).", ""]

    L += ["## 4. Drift", "",
          "- COVERED: phase definitions (drift_protocol.csv), images per phase and per-phase "
          "labelled test sets (protocol.csv; one drifted copy of the same test split per phase).",
          "- 'Offline frozen on phase 1' ~ the `static` policy, but its bank is a coreset of the "
          "first WARMUP images only, not all of phase 1.",
          "- NEW RUN: offline frozen on ALL phase-1 data; offline rebuilt on the whole stream.",
          "- **Pre-shift check FAILS** (preshift_audit.csv): PPO trains on the full stream incl. "
          "post-shift phases, the obs normaliser is fitted on a full episode, and reward fits "
          "with drifted/forget weights use post-shift labels. Needs training restricted to "
          "phase 0 — a code change, not a rerun.", ""]

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
          "- NEW RUN: measured peak GPU memory and per-image latency split into feature "
          "extraction / NN search / maintenance step (logs only hold training-time cost).", ""]

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
                  overlap, deploy, protocol, gaps, full=True):
    """full=False: per-class tables collapse to per-(dataset, budget) summaries
    so the whole report fits in one paste; full=True keeps every row."""
    S = ["=" * 78, " REVIEWER CHECKLIST — HARVESTED FROM EXISTING RUNS"
         + ("" if full else "  (short: full per-class tables in report.txt)"), "=" * 78, ""]

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
        S.append("[2] SEED VARIANCE — per-class std of stage-0 image AUROC across seeds")
        g = defaultdict(list)
        for x in per_seed:
            if x["stage"] == "0":
                g[(x["dataset"], x["budget"], x["variant"], x["policy"], x["class"])].append(x["image_auroc"])
        h = defaultdict(list)
        for (ds, b, var, pol, c), v in g.items():
            h[(ds, b, var, pol)].append((mean_std(v)[1], c, len(v)))
        S.append(box("", ["dataset", "budget", "variant", "policy", "seeds", "mean class std",
                          "max class std", "worst class"],
                     [[*k, max(n for _, _, n in v), f"{np.nanmean([sd for sd, _, _ in v]):.3f}",
                       f"{max(v)[0]:.3f}", max(v)[1]]
                      for k, v in sorted(h.items(), key=lambda kv: (kv[0][0], _budget_key(kv[0][1]), kv[0][2], kv[0][3]))]))
    else:
        S += _seed_table(per_seed)

    return _render_rest(S, per_seed, drift_rows, preshift, calib, overlap, deploy,
                        protocol, gaps, full)


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


def _render_rest(S, per_seed, drift_rows, preshift, calib, overlap, deploy, protocol, gaps, full):
    # 3. full stream: image AUROC at each quarter, cross-class mean
    S.append("[3] FULL STREAM — image AUROC at each point of the stream (cross-class mean ± std)")
    S.append("    25/50/75/100% = bank at end of that quarter, scored on THAT phase's test set;")
    S.append("    'forget@100%' = final bank scored on the stage-0 test set.")
    g = defaultdict(lambda: defaultdict(list))
    for x in per_seed:
        col = "forget@100%" if x["stage"] == FORGET else f"{int(round(100 * x['stream_frac_evaluated_at']))}%"
        g[(x["dataset"], x["backbone"], x["budget"], x["variant"], x["policy"])][col].append(x["image_auroc"])
    cols = ["25%", "50%", "75%", "100%", "forget@100%"]
    S.append(box("", ["dataset", "backbone", "budget", "variant", "policy"] + cols,
                 [[*k, *(pm(*mean_std(v[c])[:2]) for c in cols)]
                  for k, v in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1], _budget_key(kv[0][2]), kv[0][3], kv[0][4]))]))

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

    # 5. calibration
    if not full:
        S.append("[5] CALIBRATION — fits per budget (labels from the TEST split); per run in report.txt")
        bud = {}
        for x in per_seed:
            bud[x["run"]] = (x["dataset"], x["budget"], x["variant"])
        g = defaultdict(list)
        for c in calib:
            g[bud.get(c["run"], (c["dataset"], "?", "?")) + (c["target"],)].append(c)
        rows = []
        for k, v in sorted(g.items(), key=lambda kv: (kv[0][0], _budget_key(kv[0][1]), kv[0][2])):
            ps = [c["p_value"] for c in v if not isinstance(c["p_value"], str)]
            fails = [c["class"] for c in v if c["pass_p<=0.05"] is False]
            rows.append([*k, len(v), f"{np.median([c['rho_star'] for c in v]):.3f}",
                         f"{len([p for p in ps if p <= 0.05])}/{len(ps)}" if ps else "no null run",
                         f"{np.median([c['null_p95'] for c in v if c['null_p95'] != '']):.3f}" if ps else "–",
                         " ".join(fails) or "–"])
        S.append(box("", ["dataset", "budget", "variant", "target", "fits", "median ρ*",
                          "p≤.05", "median null p95", "classes failing p≤.05"], rows))
    wk = ("alpha", "beta", "gamma", "churn_coef", "churn_budget", "c90_coef", "probe_coef", "q_coef")
    if full:
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
    S.append("[7] DEPLOYMENT MEMORY — analytic fp32 MiB (latency not measured by any existing run)"
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
        protocol = keep
    S.append("[8] PROTOCOL — counts, warmup and stage-0 eval point"
             + ("" if full else " (one row per class/warmup)"))
    S.append(box("", ["run", "train", "per phase", "warmup", "eval after img", "policy steps",
                      "test normal", "test abnormal", "masks", "anomaly types"],
                 [[p["run"], p["n_train_stream"], p["images_per_phase"], p["warmup_images"],
                   p["stage0_eval_after_image"], p["stage0_policy_steps"], p["test_normal"],
                   p["test_abnormal"], p["test_nonempty_masks"], p["anomaly_types"]] for p in protocol]))
    S.append("    Method details (E, MLP, decoding, eviction, τ κ |Z| window W B, lr, episodes): hyperparams.json\n")

    S += ["=" * 78, gaps]
    return "\n".join(S)


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
        "drift": r["drift"], "suffix": r["suffix"], "capacity": r.get("capacity"),
        "warmup": r.get("warmup"), "stage0_only": r.get("stage0_only"), "job": r.get("job"),
        "gpu": r["log"].get("gpu"), "has_results": os.path.exists(os.path.join(r["dir"], "results.csv")),
        "has_fit": bool(r["fit"]), "checkpoint": r["ckpt"] or "",
        "cache_found": bool(r["cache"])} for r in runs])

    per_seed = harvest_per_seed(runs)
    write_csv(os.path.join(out, "per_seed.csv"), per_seed)
    sweep = summarize(per_seed, ["dataset", "backbone", "capacity", "class", "policy"],
                      lambda x: x["stage"] == "0")
    write_csv(os.path.join(out, "memory_sweep.csv"), sweep)
    # cross-class mean: each (class, seed) pair is one sample
    ds_mean = summarize([{**x, "seed": (x["class"], x["seed"])} for x in per_seed],
                        ["dataset", "backbone", "budget", "variant", "policy"],
                        lambda x: x["stage"] == "0")
    for d in ds_mean:
        d["n_class_seed_pairs"] = d.pop("n_seeds")
    write_csv(os.path.join(out, "memory_sweep_dataset_mean.csv"), ds_mean)
    stock = harvest_stock(pkg)
    write_csv(os.path.join(out, "stock_patchcore.csv"), stock)
    write_csv(os.path.join(out, "seed_variance.csv"), sorted(
        ({k: x[k] for k in ("dataset", "class", "backbone", "capacity", "run", "policy",
                            "seed", "image_auroc", "pixel_auroc", "pro", "writes_admit")}
         for x in per_seed if x["stage"] == "0"),
        key=lambda d: (d["run"], d["policy"], d["seed"])))
    write_csv(os.path.join(out, "fullstream.csv"), summarize(
        per_seed, ["dataset", "class", "backbone", "budget", "variant", "capacity", "policy",
                   "stream_frac_evaluated_at", "test_set"], lambda x: True))
    drift_rows, preshift = harvest_drift_protocol(), harvest_preshift(runs)
    write_csv(os.path.join(out, "drift_protocol.csv"), drift_rows)
    write_csv(os.path.join(out, "preshift_audit.csv"), preshift)
    calib = harvest_calibration(runs)
    write_csv(os.path.join(out, "calibration.csv"), calib)
    deploy, protocol = harvest_deployment(runs, pkg), harvest_protocol(runs, a.dataset_root)
    write_csv(os.path.join(out, "deployment_memory.csv"), deploy)
    write_csv(os.path.join(out, "protocol.csv"), protocol)

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

    gaps = gaps_report(runs, per_seed, calib, overlap, stock, n_logs)
    with open(os.path.join(out, "GAPS.md"), "w") as f:
        f.write(gaps)
    args = (ds_mean, sweep, stock, per_seed, drift_rows, preshift, calib,
            overlap, deploy, protocol, gaps)
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
