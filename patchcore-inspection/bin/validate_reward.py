"""Validate candidate label-free rewards against labelled AUROC — no training.

RECORD (GPU): run a fixed 15-rule panel that differs in write rate, admitted
CONTENT (random / most-novel / most-typical) and eviction, on a shared
held-out set H per seed. Saves, per run:
    boundaries  at every stage boundary: per-test-image bank scores for EVERY
                stage's labelled test set (+ labels once), per-patch scores of
                the seen held-out images, teacher S_F, synthetic-anomaly block
                scores per c, block indices, bank provenance by stage of origin,
                admits/evicts during the phase
    ckpts       6 within-phase checkpoints: held-out patch + synthetic scores,
                S_F (what an RL return actually sums over)
Synthetic anomaly (an a-priori assumption, fixed before any result): a 5%
block of a held-out image's patches displaced by c * s0 along one random unit
direction, c in {0.5, 1, 2, 4}, s0 = median warmup NN-1 distance.

ANALYZE (CPU): candidate rewards and the pre-registered statistics per dataset
-> <out>/summary.json.  VERDICT: applies the pre-registered PASS/KILL criteria
across datasets (selection: Kvasir + MVTec; confirmation: HyperKvasir).

    python bin/validate_reward.py record --cache_dir cache/kvasir_polyppvt_staged_abrupt_4 \\
        --capacity 2000 --warmup 100 --seeds 0,1,2,3 --out results/reward_validation/kvasir
    python bin/validate_reward.py analyze --out results/reward_validation/kvasir
    python bin/validate_reward.py verdict --kvasir results/reward_validation/kvasir \\
        --hyperkvasir ... --mvtec dir1,dir2,dir3
"""
import glob
import itertools
import json
import math
import os
import pickle
import sys
import time

import click
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# name -> (rate k, content, eviction); None = library baseline
PANEL = {
    "static": (0, None, None),
    "random1": (1, "random", "random"),
    "random2": (2, "random", "random"),
    "random8": (8, "random", "random"),
    "typical1_random": (1, "typical", "random"),
    "novel1_oldest": (1, "novel", "oldest"),
    "novel1_random": (1, "novel", "random"),
    "novel1_redund": (1, "novel", "redundant"),
    "novel8_oldest": (8, "novel", "oldest"),
    "novel8_random": (8, "novel", "random"),
    "novel8_redund": (8, "novel", "redundant"),
    "reservoir": None,
    "periodic_coreset": None,
    "streaming_greedy_coreset": None,
    "kcenter": None,
}
SYNTH_C = (0.5, 1.0, 2.0, 4.0)   # log-uniform defect amplitudes around s0, fixed a priori
SYNTH_FRAC = 0.05                # block = 5% of an image's patches
KNN_TIME = 10                    # time-local comparison set: 10 nearest held-out images
CANDIDATES = ("iota_mean", "iota_cvar90", "P1_lsa", "P2_lgmd", "P3_synth_int")
CONSTANTS = {"P2_lgmd": 1, "P1_lsa": 2, "P3_synth_int": 3}   # tie-break: fewer constants first


# ---- record ------------------------------------------------------------------
class FixedRule:
    """k admissions per image (random / most-novel / most-typical content);
    random, oldest or most-redundant eviction, only as many as needed."""

    def __init__(self, name, k, content, evict, seed):
        self.name, self.k, self.content, self.evict = name, k, content, evict
        self.rng = np.random.default_rng(seed)

    def step(self, env):
        from patchcore.streaming.policies import _evict_by_key

        nn = env.batch_novelty
        n = len(nn)
        if self.k == 0 or n == 0:
            admit = np.empty(0, np.int64)
        elif self.content == "novel":
            admit = np.argsort(-nn)[: self.k]
        elif self.content == "typical":
            admit = np.argsort(nn)[: self.k]
        else:
            admit = self.rng.choice(n, size=min(self.k, n), replace=False)
        n_evict = max(0, len(admit) - (env.capacity - len(env.bank)))
        slots = env.bank.active_slots()
        if n_evict == 0:
            evict = np.empty(0, np.int64)
        elif self.evict == "random":
            evict = self.rng.choice(slots, size=min(n_evict, len(slots)), replace=False)
        elif self.evict == "oldest":
            evict = _evict_by_key(env, n_evict, env.bank.insert_step[slots])
        else:
            evict = _evict_by_key(env, n_evict, env.bank.entry_features()[:, 2])
        return env.step_with_decision(admit, evict)


def _policy(name, seed):
    from patchcore.streaming import policies as P

    spec = PANEL[name]
    if spec is None:
        cls = P.BASELINES[name]
        return cls(seed=seed) if name in ("reservoir", "periodic_coreset") else cls()
    return FixedRule(name, *spec, seed=seed)


def _patch_scores(bank, q, k):
    """per row of q: mean of the k smallest squared L2 distances to the bank."""
    from patchcore.streaming.env import _bank_topk_sq

    if len(q) == 0:
        return np.zeros(0, np.float32)
    d, _ = _bank_topk_sq(bank, q, (q * q).sum(1), k)
    return torch.nan_to_num(d, posinf=float("nan")).nanmean(1).cpu().numpy()


def _image_scores(bank, reader, k, chunk_images=64):
    """Test-image scores exactly as evaluate_bank_on_stage (k-NN mean of
    squared L2, max over patches), streamed in image chunks."""
    from patchcore.streaming.bank import knn_device

    dev = knn_device()
    base = torch.from_numpy(bank.vectors()).to(dev)
    kk = min(k, len(base))
    emb = reader.embeddings
    n, p, d = emb.shape
    s = np.empty(n)
    for i in range(0, n, chunk_images):
        q = torch.from_numpy(np.ascontiguousarray(emb[i:i + chunk_images], dtype=np.float32))
        q = q.to(dev).reshape(-1, d)
        pd = (torch.cdist(q, base) ** 2).topk(kk, dim=1, largest=False).values.mean(1)
        s[i:i + chunk_images] = pd.reshape(-1, p).max(1).values.double().cpu().numpy()
    return s


def _synth_block(stream, n, seed, s0, patch_shape):
    """Deterministic per (seed, n): block indices and {c: displaced patches}."""
    x = np.asarray(stream.image_patches(n), dtype=np.float32)
    p, d = x.shape
    rng = np.random.default_rng([seed, int(n), 7])
    m = max(1, int(round(SYNTH_FRAC * p)))
    if patch_shape and patch_shape[0] * patch_shape[1] == p:
        h, w = patch_shape
        side = max(1, int(round(np.sqrt(m))))
        r0, c0 = rng.integers(0, h - side + 1), rng.integers(0, w - side + 1)
        idx = np.array([(r0 + i) * w + c0 + j for i in range(side) for j in range(side)])
    else:
        idx = rng.choice(p, size=m, replace=False)
    u = rng.normal(size=d).astype(np.float32)
    u /= np.linalg.norm(u) + 1e-8
    return idx, {c: x[idx] + u * (c * s0) for c in SYNTH_C}


@click.group()
def cli():
    pass


@cli.command()
@click.option("--cache_dir", required=True)
@click.option("--synthetic_tests", default=None,
              help="fixture only: SyntheticConfig kwargs, e.g. 'seed=0,patches_per_image=64,"
                   "images_per_stage=150,anomaly_dist=1.5' (cache has no test sets)")
@click.option("--capacity", type=int, required=True)
@click.option("--warmup", type=int, default=100)
@click.option("--n_nn", type=int, default=5)
@click.option("--heldout_frac", type=float, default=0.05)
@click.option("--seeds", default="0,1,2,3")
@click.option("--policies", default=",".join(PANEL))
@click.option("--ckpts_per_phase", type=int, default=6)
@click.option("--out", required=True)
def record(cache_dir, synthetic_tests, capacity, warmup, n_nn, heldout_frac, seeds, policies,
           ckpts_per_phase, out):
    from patchcore.streaming.bank import device_banner, knn_device
    from patchcore.streaming.cache import EmbeddingCacheReader
    from patchcore.streaming.env import MemoryMaintenanceEnv
    from patchcore.streaming.teacher import load_teacher

    print(device_banner())
    stream = EmbeddingCacheReader(os.path.join(cache_dir, "stream"))
    teach = load_teacher(os.path.join(cache_dir, f"teacher_k{n_nn}.npz"))
    if synthetic_tests:
        from patchcore.streaming.synthetic import SyntheticConfig, make_all_test_stages

        kw = {k: type(getattr(SyntheticConfig, k))(v)
              for k, v in (a.split("=") for a in synthetic_tests.split(","))}
        tests = make_all_test_stages(SyntheticConfig(**kw))
    else:
        tdir = os.path.join(cache_dir, "test")
        tests = [EmbeddingCacheReader(os.path.join(tdir, s))
                 for s in sorted(d for d in os.listdir(tdir) if d.startswith("stage_"))]
    labels = [np.asarray(r.labels, int) for r in tests]
    patch_shape = tuple(stream.manifest.get("patch_shape", ()) or ()) or None
    stage_ids = np.array([stream.stage_of(i) for i in range(stream.n_images)])
    n_stages = int(stage_ids.max()) + 1
    phase_len = int(np.median(np.bincount(stage_ids)))
    ckpt_every = max(1, math.ceil(phase_len / ckpts_per_phase))
    os.makedirs(out, exist_ok=True)
    dev = knn_device()
    json.dump({"cache_dir": cache_dir, "capacity": capacity, "warmup": warmup, "n_nn": n_nn,
               "heldout_frac": heldout_frac, "synth_c": SYNTH_C, "synth_frac": SYNTH_FRAC,
               "ckpt_every": ckpt_every}, open(os.path.join(out, "record_config.json"), "w"))

    for seed in (int(s) for s in seeds.split(",")):
        # a-priori scale from a throwaway env; policy envs each reset once, so
        # every policy sees the same held-out set H for this seed
        scale_env = MemoryMaintenanceEnv(
            stream, capacity=capacity, warmup_images=warmup, seed=seed,
            n_nearest_neighbours=n_nn, reward_mode="iota", teacher=teach,
            heldout_frac=heldout_frac, heldout_seed=seed, compute_obs=False)
        scale_env.reset()
        probe = torch.from_numpy(np.asarray(scale_env._probe, np.float32)).to(dev)
        s0 = float(np.median(np.sqrt(_patch_scores(scale_env.bank, probe, 1))))
        synth = {int(n): _synth_block(stream, n, seed, s0, patch_shape)
                 for n in scale_env.heldout_ids}
        print(f"[seed {seed}] s0={s0:.4g} |H|={len(synth)} ckpt every {ckpt_every} images",
              flush=True)

        for name in policies.split(","):
            path = os.path.join(out, f"{name}_s{seed}.pkl")
            if os.path.exists(path):
                try:
                    old = pickle.load(open(path, "rb"))
                except Exception:
                    old = {}
                if "ckpts" in old:  # current format; older recordings are re-recorded
                    print(f"[skip] {path} exists")
                    continue
            t_start = time.time()
            env = MemoryMaintenanceEnv(
                stream, capacity=capacity, warmup_images=warmup, seed=seed,
                n_nearest_neighbours=n_nn, reward_mode="iota", teacher=teach,
                heldout_frac=heldout_frac, heldout_seed=seed, compute_obs=False)
            pol = _policy(name, seed)
            boundaries, ckpts = [], []
            phase_io = [0, 0]

            def snap(env, full):
                t = env._t - 1                          # last processed image
                q, _, _, k = env.heldout_pool_seen()
                ids = env.heldout_ids[env.heldout_ids <= t]   # S_F[n, t] needs n <= t
                ppi = len(q) // max(1, len(env.heldout_seen_ids()))
                ps = _patch_scores(env.bank, q[: len(ids) * ppi], k).reshape(len(ids), -1)
                sy = {}
                for c in SYNTH_C:
                    if len(ids):
                        x = torch.from_numpy(np.concatenate([synth[int(n)][1][c] for n in ids]))
                        sy[c] = _patch_scores(env.bank, x.to(dev), k).reshape(len(ids), -1)
                    else:
                        sy[c] = np.zeros((0, 1), np.float32)
                rec = {"t": int(t), "ids": ids, "id_stage": stage_ids[ids],
                       "patch_scores": ps.astype(np.float16),
                       "S_F": teach["S_F"][ids, t].astype(np.float32),
                       "synth": {c: v.astype(np.float16) for c, v in sy.items()}}
                if full:
                    act = env.bank.active_slots()
                    rec.update(
                        synth_idx=[synth[int(n)][0] for n in ids],
                        test_scores=[_image_scores(env.bank, r, n_nn).astype(np.float32)
                                     for r in tests],
                        provenance=np.bincount(env.bank.insert_stage[act], minlength=n_stages),
                        phase_admits=phase_io[0], phase_evicts=phase_io[1])
                return rec

            env.reset()
            done, steps = False, 0
            while not done:
                stage_before = env.stage
                _, _, done, info = pol.step(env)
                phase_io[0] += info["n_admit"]
                phase_io[1] += info["n_evict"]
                steps += 1
                if done or env.stage != stage_before:
                    b = snap(env, True)
                    b["stage"] = int(stage_before)
                    boundaries.append(b)
                    phase_io[:] = [0, 0]
                elif steps % ckpt_every == 0 and (env.heldout_ids <= env._t - 1).any():
                    ckpts.append(snap(env, False))
            rec = {"policy": name, "seed": seed, "eps": float(teach["eps"]), "s0": s0,
                   "test_labels": labels, "boundaries": boundaries, "ckpts": ckpts}
            with open(path, "wb") as f:
                pickle.dump(rec, f)
            au = [round(_auroc(b["test_scores"][b["stage"]], labels[b["stage"]]), 3)
                  for b in boundaries]
            print(f"{name:26s} s{seed} stage AUROC {au} forget "
                  f"{_auroc(boundaries[-1]['test_scores'][0], labels[0]):.3f} "
                  f"({time.time() - t_start:.0f}s)", flush=True)


# ---- candidates (label-free; higher = better bank) ------------------------------
def _auroc(scores, y):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, scores))


def _soft_auc_pairs(a, n, mask):
    """mean over masked (anomaly i, normal j) pairs of [a_i > n_j] (+1/2 ties)."""
    gt = (a[:, None] > n[None]) + 0.5 * (a[:, None] == n[None])
    return float((gt * mask).sum() / max(1, mask.sum()))


def candidates(snap, eps):
    """ι mean / CVaR (controls) and the round-1 proposals P1–P3."""
    sb = np.nanmax(snap["patch_scores"].astype(np.float64), axis=1)
    out = {}
    if len(sb) == 0:
        return {c: float("nan") for c in CANDIDATES}
    lr = np.log(np.maximum(sb, eps) / np.maximum(snap["S_F"], eps))
    tail = np.sort(lr)[-max(1, int(np.ceil(0.1 * len(lr)))):]
    out["iota_mean"] = -float(lr.mean())
    out["iota_cvar90"] = -float(tail.mean())
    anom = {c: np.maximum(sb, np.nanmax(v.astype(np.float64), axis=1))
            for c, v in snap["synth"].items()}
    full = np.ones((len(sb), len(sb)), bool)
    for c, a in anom.items():
        out[f"synth_c{c:g}"] = _soft_auc_pairs(a, sb, full)
    # P3: pooled synthetic AUROC integrated over the c grid
    out["P3_synth_int"] = float(np.mean([out[f"synth_c{c:g}"] for c in anom]))
    # time-local comparison set: self + KNN_TIME nearest held-out images in
    # stream time (no phase labels)
    tid = np.asarray(snap["ids"], float)
    kk = min(len(tid), KNN_TIME + 1)
    nb = np.argsort(np.abs(tid[:, None] - tid[None]), axis=1, kind="stable")[:, :kk]
    mask = np.zeros((len(tid), len(tid)), bool)
    np.put_along_axis(mask, nb, True, axis=1)
    # P1: time-local synthetic AUROC, c-integrated
    out["P1_lsa"] = float(np.mean([_soft_auc_pairs(a, sb, mask) for a in anom.values()]))
    # P2: time-local Gini mean difference of normal scores (negated: less spread = better)
    m2 = mask.copy()
    np.fill_diagonal(m2, False)
    out["P2_lgmd"] = -float((np.abs(sb[:, None] - sb[None]) * m2).sum() / max(1, m2.sum()))
    return out


# ---- analyze --------------------------------------------------------------------
def _boot_auroc(scores, y, neg_idx, pos_idx):
    """AUROC for every bootstrap row (rank-based, ordinal ties — continuous scores)."""
    s = np.concatenate([scores[neg_idx], scores[pos_idx]], axis=1)
    r = s.argsort(axis=1).argsort(axis=1) + 1.0
    n_neg, n_pos = neg_idx.shape[1], pos_idx.shape[1]
    return (r[:, n_neg:].sum(1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


@cli.command()
@click.option("--out", required=True)
@click.option("--nboot", type=int, default=1000)
def analyze(out, nboot):
    from scipy import stats

    recs = [pickle.load(open(p, "rb")) for p in sorted(glob.glob(os.path.join(out, "*_s*.pkl")))]
    if not recs:
        raise click.UsageError(f"no recordings in {out}")
    pols = sorted({r["policy"] for r in recs})
    seeds = sorted({r["seed"] for r in recs})
    R = {(r["policy"], r["seed"]): r for r in recs}
    n_st = len(recs[0]["boundaries"])
    labels = recs[0]["test_labels"]

    # paired bootstrap: the SAME resampled test images for every policy (per seed, stage)
    boot_idx = {}
    for sd in seeds:
        for u in range(n_st):
            y = labels[u]
            rng = np.random.default_rng([sd, u, 99])
            neg, pos = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
            boot_idx[(sd, u)] = (neg[rng.integers(0, len(neg), (nboot, len(neg)))],
                                 pos[rng.integers(0, len(pos), (nboot, len(pos)))])

    cell = {}   # (policy, seed, boundary) -> stats
    for (p, sd), r in R.items():
        for b in r["boundaries"]:
            s = b["stage"]
            au = np.array([_auroc(b["test_scores"][u], labels[u]) for u in range(n_st)])
            boots = np.stack([_boot_auroc(b["test_scores"][u], labels[u], *boot_idx[(sd, u)])
                              for u in range(s + 1)])
            cell[(p, sd, s)] = {"cur": au[s], "acc": float(au[: s + 1].mean()), "s0": au[0],
                                "boot_cur": boots[s], "boot_acc": boots.mean(0),
                                **candidates(b, r["eps"])}
    cand = list(CANDIDATES)

    # (A) decisive-pair concordance, boundaries >= 1, within seed
    outcomes = {c: {"cur": [], "acc": []} for c in cand}
    for sd in seeds:
        for s in range(1, n_st):
            for p, q in itertools.combinations(pols, 2):
                xp, xq = cell[(p, sd, s)], cell[(q, sd, s)]
                for tgt in ("cur", "acc"):
                    d = xp[f"boot_{tgt}"] - xq[f"boot_{tgt}"]
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    if lo > 0 or hi < 0:
                        true = np.sign(xp[tgt] - xq[tgt])
                        for c in cand:
                            dc = xp[c] - xq[c]
                            o = 0.5 if dc == 0 or not np.isfinite(dc) else float(np.sign(dc) == true)
                            outcomes[c][tgt].append((sd, o))

    def dpc(lst):
        if not lst:
            return float("nan"), (float("nan"), float("nan")), 0
        rng = np.random.default_rng(0)
        sd_arr = np.array([x[0] for x in lst])
        o = np.array([x[1] for x in lst])
        boots = []
        for _ in range(1000):   # 90% CI by resampling seeds
            v = np.concatenate([o[sd_arr == s] for s in rng.choice(seeds, len(seeds))])
            boots.append(v.mean() if len(v) else np.nan)
        return float(o.mean()), tuple(float(x) for x in np.nanpercentile(boots, [5, 95])), len(o)

    # (B) contrasts: sign agreement over seed x boundary cells at decisive boundaries
    contrasts = [("static", "random1", (1, 2, 3)), ("static", "random2", (1, 2, 3)),
                 ("novel1_random", "random1", (1, 2, 3)),
                 ("periodic_coreset", "static", (2, 3)), ("novel1_oldest", "static", (2, 3))]
    contrast_res = {}
    for a, b_, bset in contrasts:
        if a not in pols or b_ not in pols:
            continue
        agree = {c: [] for c in cand}
        decisive = []
        for s in bset:
            if s >= n_st:
                continue
            d_boot = np.mean([cell[(a, sd, s)]["boot_acc"] - cell[(b_, sd, s)]["boot_acc"]
                              for sd in seeds], axis=0)
            lo, hi = np.percentile(d_boot, [2.5, 97.5])
            if not (lo > 0 or hi < 0):
                continue
            truth = np.sign(np.mean([cell[(a, sd, s)]["acc"] - cell[(b_, sd, s)]["acc"]
                                     for sd in seeds]))
            decisive.append(s)
            for c in cand:
                for sd in seeds:
                    agree[c].append(float(np.sign(cell[(a, sd, s)][c] - cell[(b_, sd, s)][c]) == truth))
        contrast_res[f"{a}>{b_}"] = {
            "decisive_boundaries": decisive,
            "agreement": {c: (float(np.mean(v)) if v else None) for c, v in agree.items()}}

    # (C) anytime: time-average of the candidate vs final ACC, within seed
    anytime = {c: [] for c in cand}
    for sd in seeds:
        tav, facc = {c: [] for c in cand}, []
        for p in pols:
            r = R[(p, sd)]
            snaps = sorted(r["ckpts"] + r["boundaries"], key=lambda z: z["t"])
            vals = [candidates(z, r["eps"]) for z in snaps if len(z["ids"]) >= 3]
            for c in cand:
                tav[c].append(np.nanmean([v[c] for v in vals]) if vals else np.nan)
            last = r["boundaries"][-1]
            facc.append(np.mean([_auroc(last["test_scores"][u], labels[u]) for u in range(n_st)]))
        for c in cand:
            anytime[c].append(stats.spearmanr(tav[c], facc).correlation)

    # saturation of the synthetic family (kill rule for P1/P3)
    synth_cols = [k for k in next(iter(cell.values())) if k.startswith("synth_c")]
    vals = np.array([[cell[k][sc] for sc in synth_cols] for k in cell])
    saturated = bool(np.all((vals >= 0.999) | (vals <= 0.51)))

    # regime diagnostic r = d median log anomaly / d median log normal (vs static, final boundary)
    s = n_st - 1
    dn, da = [], []
    for p in pols:
        if p == "static":
            continue
        for sd in seeds:
            y = labels[s]
            a_ = np.log(np.maximum(R[(p, sd)]["boundaries"][-1]["test_scores"][s], 1e-12))
            b0 = np.log(np.maximum(R[("static", sd)]["boundaries"][-1]["test_scores"][s], 1e-12))
            dn.append(np.median(a_[y == 0]) - np.median(b0[y == 0]))
            da.append(np.median(a_[y == 1]) - np.median(b0[y == 1]))
    regime_r = float(np.mean(da) / np.mean(dn)) if abs(np.mean(dn)) > 1e-12 else float("nan")

    summary = {"out": out, "policies": pols, "seeds": seeds, "n_boundaries": n_st,
               "regime_r": regime_r, "synth_saturated": saturated,
               "dpc": {}, "anytime": {}, "contrasts": contrast_res,
               "pair_outcomes": {c: outcomes[c]["acc"] for c in cand}}
    print(f"== {out}: {len(pols)} policies x {len(seeds)} seeds, {n_st} boundaries; "
          f"regime r = {regime_r:.2f} (>=0.7 favours spread rewards, <=0.3 location); "
          f"synthetic family saturated: {saturated}")
    print("\n[A] decisive-pair concordance (fraction of decisive policy pairs ordered like labelled AUROC)")
    print(f"{'candidate':14s} {'ACC':>6s} {'90% CI':>14s} {'n':>5s}   {'cur':>6s} {'n':>5s}   anytime rho")
    for c in cand:
        m_acc, ci, n_acc = dpc(outcomes[c]["acc"])
        m_cur, _, n_cur = dpc(outcomes[c]["cur"])
        at = float(np.nanmean(anytime[c]))
        summary["dpc"][c] = {"acc": m_acc, "acc_ci": ci, "n_acc": n_acc, "cur": m_cur, "n_cur": n_cur}
        summary["anytime"][c] = at
        print(f"{c:14s} {m_acc:6.3f} [{ci[0]:.2f},{ci[1]:.2f}] {n_acc:5d}   {m_cur:6.3f} {n_cur:5d}   {at:+.2f}")
    print("\n[B] contrasts (sign agreement over seed x decisive-boundary cells)")
    for k, v in contrast_res.items():
        ag = " ".join(f"{c}={'-' if x is None else f'{x:.2f}'}" for c, x in v["agreement"].items())
        print(f"{k:28s} decisive b={v['decisive_boundaries']}: {ag}")
    print("\n[seed-mean] per policy at the last boundary: cur, ACC, stage0 | candidates")
    s = n_st - 1
    for p in sorted(pols, key=lambda p: -np.mean([cell[(p, sd, s)]["acc"] for sd in seeds])):
        g = [cell[(p, sd, s)] for sd in seeds]
        print(f"{p:26s} {np.mean([x['cur'] for x in g]):.3f} {np.mean([x['acc'] for x in g]):.3f} "
              f"{np.mean([x['s0'] for x in g]):.3f} | "
              + " ".join(f"{c}={np.mean([x[c] for x in g]):+.4f}" for c in cand))
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(f"\nwrote {os.path.join(out, 'summary.json')}")


@cli.command()
@click.option("--kvasir", required=True, help="Kvasir result dir (selection)")
@click.option("--mvtec", default="", help="comma-separated MVTec class dirs (selection, pooled)")
@click.option("--hyperkvasir", default="", help="HyperKvasir dir (confirmation)")
def verdict(kvasir, mvtec, hyperkvasir):
    """Pre-registered PASS/KILL criteria (critic round 1)."""
    load = lambda d: json.load(open(os.path.join(d, "summary.json")))  # noqa: E731
    K = load(kvasir)
    H = load(hyperkvasir) if hyperkvasir else None
    Ms = [load(d) for d in mvtec.split(",") if d]

    def pooled_dpc(summaries, c):
        o = [x[1] for s in summaries for x in s["pair_outcomes"][c]]
        return (float(np.mean(o)) if o else float("nan")), len(o)

    rows = []
    for c in CANDIDATES:
        why_kill, why_fail = [], []
        k_dpc = K["dpc"][c]["acc"]
        if k_dpc < 0.65:
            why_kill.append(f"Kvasir DPC {k_dpc:.2f} < .65")
        a_sr = K["contrasts"].get("static>random1", {}).get("agreement", {}).get(c)
        if a_sr is not None and a_sr < 0.5:
            why_kill.append(f"static>random1 agreement {a_sr:.2f} < .5")
        if K["anytime"][c] < 0.3:
            why_kill.append(f"Kvasir anytime {K['anytime'][c]:+.2f} < .3")
        if c in ("P1_lsa", "P3_synth_int") and K["synth_saturated"]:
            why_kill.append("synthetic family saturated")
        if H and H["dpc"][c]["acc"] < 0.65:
            why_kill.append(f"HyperKvasir DPC {H['dpc'][c]['acc']:.2f} < .65")
        if k_dpc < 0.75:
            why_fail.append(f"A: Kvasir DPC {k_dpc:.2f} < .75")
        if H and H["dpc"][c]["acc"] < 0.75:
            why_fail.append(f"A: HyperKvasir DPC {H['dpc'][c]['acc']:.2f} < .75")
        m_dpc, m_n = pooled_dpc(Ms, c) if Ms else (float("nan"), 0)
        if Ms and m_n >= 30 and m_dpc < 0.65:
            why_fail.append(f"A: MVTec DPC {m_dpc:.2f} < .65 (n={m_n})")
        for name, v in K["contrasts"].items():
            x = v["agreement"].get(c)
            if x is not None and x < 2 / 3:
                why_fail.append(f"B: {name} {x:.2f} < 2/3")
        if K["anytime"][c] < 0.5:
            why_fail.append(f"C: Kvasir anytime {K['anytime'][c]:+.2f} < .5")
        if H and H["anytime"][c] < 0.5:
            why_fail.append(f"C: HyperKvasir anytime {H['anytime'][c]:+.2f} < .5")
        if c != "iota_mean" and k_dpc - K["dpc"]["iota_mean"]["acc"] < 0.10:
            why_fail.append(f"D: Kvasir DPC - iota {k_dpc - K['dpc']['iota_mean']['acc']:+.2f} < .10")
        sel, _ = pooled_dpc([K] + Ms, c)
        status = "KILL" if why_kill else ("PASS" if not why_fail else "FAIL")
        rows.append((c, status, sel, why_kill + why_fail))
    print("=== pre-registered verdict (selection: Kvasir + MVTec pooled DPC; confirmation: HyperKvasir) ===")
    for c, st, sel, why in rows:
        print(f"{c:14s} {st:5s} selection DPC {sel:.3f}  " + ("; ".join(why) if why else "all criteria met"))
    passing = sorted([r for r in rows if r[1] == "PASS" and r[0].startswith("P")], key=lambda r: -r[2])
    if not passing:
        print("\nNo candidate passes -> report a negative result; a new candidate needs a fresh dataset.")
        return
    close = [r for r in passing if passing[0][2] - r[2] <= 0.02]
    winner = min(close, key=lambda r: CONSTANTS.get(r[0], 9))
    print(f"\nWINNER: {winner[0]} (selection DPC {winner[2]:.3f})")


if __name__ == "__main__":
    cli()
