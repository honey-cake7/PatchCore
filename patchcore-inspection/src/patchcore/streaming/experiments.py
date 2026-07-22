"""Gate experiments: headroom (Gate 1) and proxy validation (Gate 2).

Gate 1 asks *is there anything to fix* — does a static stage-0 bank lose
accuracy as the stream drifts, while an oracle per-stage-retrained bank holds?
Gate 2 asks *is the label-free proxy trustworthy* — does the proxy reward
(coverage + redundancy) rank banks the same way labeled AUROC/PRO does? Both
gates must pass before any policy learning.
"""
from typing import Callable, Dict, List, Optional

import numpy as np

import patchcore.common
import patchcore.sampler
from patchcore.streaming.bank import DynamicMemoryBank
from patchcore.streaming.evaluate import evaluate_bank_on_stage
from patchcore.streaming.reward import coverage, estimate_scales, redundancy


# ---- shared helpers ------------------------------------------------------
def stage_patches(reader, stage: int, max_images: Optional[int] = None) -> np.ndarray:
    ids = [i for i in range(reader.n_images) if reader.stage_of(i) == stage]
    if max_images is not None:
        ids = ids[:max_images]
    return reader.flat_slice(ids)


def coreset_bank(
    features: np.ndarray, capacity: int, device=None, seed: int = 0
) -> DynamicMemoryBank:
    """Greedy-coreset ``features`` down to ``capacity`` and load into a bank.

    ``device=None`` auto-selects the k-NN device (GPU when available) — the
    greedy coreset is by far the slowest part of the gates and the
    periodic-coreset baseline at large capacities.
    """
    import torch

    from patchcore.streaming.bank import knn_device

    device = torch.device(device) if device else knn_device()
    features = np.ascontiguousarray(features, dtype=np.float32)
    if len(features) > capacity:
        pct = capacity / len(features)
        sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(
            percentage=pct, device=device
        )
        from patchcore.streaming.bank import pinned_global_seed

        with pinned_global_seed(seed):
            features = sampler.run(features)
    return DynamicMemoryBank.from_vectors(features[:capacity], capacity=capacity, seed=seed)


# ---- Gate 1: headroom ----------------------------------------------------
def run_headroom(
    stream_reader,
    test_readers: List,
    capacity: int,
    n_nearest_neighbours: int = 1,
    patch_shape=None,
    imagesize=None,
    device: Optional[str] = None,
    max_images_per_stage: Optional[int] = None,
) -> List[Dict]:
    """Static stage-0 bank vs per-stage oracle coreset, evaluated per stage.

    Pass condition (checked by the caller): static AUROC/PRO drops materially at
    later stages while the oracle holds — that gap is the headroom the learned
    policy can recover.
    """
    from patchcore.streaming.bank import knn_device

    device = str(device) if device else str(knn_device())
    n_stages = len(test_readers)
    static = coreset_bank(
        stage_patches(stream_reader, 0, max_images_per_stage), capacity, device
    )
    results = []
    for stage in range(n_stages):
        static_m = evaluate_bank_on_stage(
            static, test_readers[stage], n_nearest_neighbours, patch_shape,
            imagesize, device,
        )
        oracle = coreset_bank(
            stage_patches(stream_reader, stage, max_images_per_stage), capacity, device
        )
        oracle_m = evaluate_bank_on_stage(
            oracle, test_readers[stage], n_nearest_neighbours, patch_shape,
            imagesize, device,
        )
        results.append(
            {
                "stage": stage,
                "static_auroc": static_m["image_auroc"],
                "oracle_auroc": oracle_m["image_auroc"],
                "static_pro": static_m.get("pro", float("nan")),
                "oracle_pro": oracle_m.get("pro", float("nan")),
            }
        )
    return results


def headroom_gap(results: List[Dict]) -> float:
    """Mean oracle-minus-static AUROC gap over the drifted (stage>0) stages."""
    gaps = [
        r["oracle_auroc"] - r["static_auroc"]
        for r in results
        if r["stage"] > 0 and np.isfinite(r["static_auroc"])
    ]
    return float(np.mean(gaps)) if gaps else 0.0


# ---- Gate 2: proxy validation -------------------------------------------
def generate_bank_states(
    stream_reader, capacity: int, device: Optional[str] = None, seed: int = 0
) -> List[DynamicMemoryBank]:
    """A diverse spread of bank states spanning good→bad coverage/redundancy.

    Includes per-stage coresets, mixed-stage coresets, random subsets at several
    sizes, single-stage subsets, and deliberately clumped (redundant) banks.
    """
    rng = np.random.default_rng(seed)
    n_stages = int(max(stream_reader.stage_of(i) for i in range(stream_reader.n_images)) + 1)
    all_ids = list(range(stream_reader.n_images))
    mixed = stream_reader.flat_slice(rng.choice(all_ids, size=min(len(all_ids), 400), replace=False))

    states: List[DynamicMemoryBank] = []
    # per-stage coresets (good for their own stage)
    for s in range(n_stages):
        feats = stage_patches(stream_reader, s, max_images=120)
        if len(feats):
            states.append(coreset_bank(feats, capacity, device, seed=s))
    # mixed-stage coreset
    states.append(coreset_bank(mixed, capacity, device, seed=100))
    # random subsets at several sizes from the mixed pool
    for frac in (0.15, 0.35, 0.6, 1.0):
        k = max(4, int(capacity * frac))
        idx = rng.choice(len(mixed), size=min(k, len(mixed)), replace=False)
        states.append(DynamicMemoryBank.from_vectors(mixed[idx], capacity=capacity, seed=int(frac * 100)))
    # single-stage random subsets (poor coverage of other stages)
    for s in range(n_stages):
        feats = stage_patches(stream_reader, s, max_images=60)
        if len(feats):
            idx = rng.choice(len(feats), size=min(capacity, len(feats)), replace=False)
            states.append(DynamicMemoryBank.from_vectors(feats[idx], capacity=capacity, seed=s + 7))
    # deliberately clumped banks: few distinct points replicated (high redundancy)
    for n_distinct in (5, 20):
        base = mixed[rng.choice(len(mixed), size=n_distinct, replace=False)]
        reps = int(np.ceil(capacity / n_distinct))
        clumped = np.repeat(base, reps, axis=0)[:capacity]
        clumped = clumped + rng.normal(scale=1e-3, size=clumped.shape).astype(np.float32)
        states.append(DynamicMemoryBank.from_vectors(clumped, capacity=capacity, seed=n_distinct))
    return states


def record_policy_traces(
    stream_reader,
    test_readers: List,
    capacity: int,
    warmup: int = 100,
    n_nearest_neighbours: int = 1,
    patch_shape=None,
    imagesize=None,
    policy_names: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
) -> List[Dict]:
    """Drive each baseline policy over the stream, recording per-step reward
    components alongside labeled per-stage AUROC and final forgetting.

    One trace per (policy, seed). Because the proxy reward is linear in its
    components, any candidate weighting can later be scored on these traces
    offline (``fit_reward_weights``) — no policy re-runs per candidate.
    """
    from patchcore.streaming import policies as P
    from patchcore.streaming.env import MemoryMaintenanceEnv

    policy_names = policy_names or list(P.BASELINES)
    seeds = seeds if seeds is not None else [0]

    def eval_fn(env, stage):
        return evaluate_bank_on_stage(
            env.bank, test_readers[stage], n_nearest_neighbours, patch_shape,
            imagesize,
        )

    traces = []
    for name in policy_names:
        cls = P.BASELINES[name]
        for seed in seeds:
            env = MemoryMaintenanceEnv(
                stream_reader, capacity=capacity, warmup_images=warmup,
                seed=seed, n_nearest_neighbours=n_nearest_neighbours,
            )
            policy = cls(k=8) if name in ("fifo", "random") else cls()
            summ = P.run_policy(env, policy, per_stage_eval=eval_fn)
            forget_m = evaluate_bank_on_stage(
                env.bank, test_readers[0], n_nearest_neighbours, patch_shape,
                imagesize,
            )
            infos = summ["infos"]
            traces.append({
                "policy": name,
                "seed": int(seed),
                "C": np.asarray([i["C"] for i in infos], dtype=np.float64),
                "R": np.asarray([i["R"] for i in infos], dtype=np.float64),
                "churn": np.asarray([i["churn"] for i in infos], dtype=np.float64),
                "score_drift": np.asarray(
                    [i["score_drift"] for i in infos], dtype=np.float64
                ),
                "C90": np.asarray([i["C90"] for i in infos], dtype=np.float64),
                "P": np.asarray([i["P"] for i in infos], dtype=np.float64),
                "stage": np.asarray([i["stage"] for i in infos], dtype=np.int64),
                "stage_aurocs": {
                    int(ev["stage"]): float(ev["image_auroc"]) for ev in summ["evals"]
                },
                "forget_auroc": float(forget_m["image_auroc"]),
            })
            print(f"[trace] {name:26s} seed={seed} "
                  f"aurocs={[round(v, 3) for _, v in sorted(traces[-1]['stage_aurocs'].items())]} "
                  f"forget={traces[-1]['forget_auroc']:.3f}")
    return traces


def fit_reward_weights(
    traces: List[Dict],
    alphas=(0.0, 0.25, 1.0),
    betas=(0.0, 0.15, 0.3, 0.6, 1.0),
    gammas=(0.0, 0.1, 0.3),
    churn_coefs=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
    churn_budgets=(0.0, 0.005, 0.01, 0.02),
    c90_coefs=(0.0, 0.25, 0.5, 1.0, 2.0),
    probe_coefs=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
    q_coefs=(0.0, 1.0, 2.0, 4.0, 8.0),
    forget_weight: float = 0.5,
) -> Dict:
    """Grid-fit reward weights so mean episode reward ranks policies like AUROC.

    Target per trace: mean drifted-stage (stage>=1) image AUROC plus
    ``forget_weight`` times the stage-0 forgetting AUROC — the two quantities
    the learned policy is meant to maximize. Reports the Spearman rho of the
    best candidate and of the current production ``RewardConfig`` defaults
    (the misalignment baseline), plus the top-5 candidates so a flat optimum
    (many near-ties) is visible.
    """
    from scipy import stats

    from patchcore.streaming.reward import RewardConfig

    targets = []
    mean_c, mean_r, mean_sd, mean_c90, mean_p, mean_q = [], [], [], [], [], []
    for tr in traces:
        drifted = [v for s, v in tr["stage_aurocs"].items() if s >= 1]
        targets.append(float(np.mean(drifted)) + forget_weight * tr["forget_auroc"])
        mean_c.append(tr["C"].mean())
        mean_r.append(tr["R"].mean())
        mean_sd.append(tr["score_drift"].mean())
        # older traces lack C90/P; treat as absent (coef grid still explores 0)
        mean_c90.append(tr["C90"].mean() if "C90" in tr else 0.0)
        mean_p.append(tr["P"].mean() if "P" in tr else 0.0)
        # scale-invariant tail ratio, derivable from any trace that has C90
        mean_q.append(
            float(np.mean(tr["C90"] / np.maximum(tr["C"], 1e-8)))
            if "C90" in tr else 0.0
        )
    targets = np.asarray(targets)
    mean_c = np.asarray(mean_c)
    mean_r = np.asarray(mean_r)
    mean_sd = np.asarray(mean_sd)
    mean_c90 = np.asarray(mean_c90)
    mean_p = np.asarray(mean_p)
    mean_q = np.asarray(mean_q)
    # mean excess churn per trace, per candidate budget
    mean_excess = {
        b: np.asarray([np.maximum(tr["churn"] - b, 0.0).mean() for tr in traces])
        for b in churn_budgets
    }

    def rho_for(alpha, beta, gamma, coef, budget, c90c, probec, qc):
        proxy = -(alpha * mean_c + beta * mean_r + gamma * mean_sd
                  + coef * mean_excess[budget]
                  + c90c * mean_c90 + probec * mean_p + qc * mean_q)
        return float(stats.spearmanr(proxy, targets).correlation)

    candidates = []
    for alpha in alphas:
        for beta in betas:
            for gamma in gammas:
                for coef in churn_coefs:
                    for budget in churn_budgets:
                        for c90c in c90_coefs:
                            for probec in probe_coefs:
                                for qc in q_coefs:
                                    rho = rho_for(alpha, beta, gamma, coef,
                                                  budget, c90c, probec, qc)
                                    if not np.isfinite(rho):
                                        continue
                                    candidates.append({
                                        "alpha": float(alpha),
                                        "beta": float(beta),
                                        "gamma": float(gamma),
                                        "churn_coef": float(coef),
                                        "churn_budget": float(budget),
                                        "c90_coef": float(c90c),
                                        "probe_coef": float(probec),
                                        "q_coef": float(qc),
                                        "rho_ranking": rho,
                                    })
    candidates.sort(key=lambda d: -d["rho_ranking"])
    best = candidates[0]

    cur = RewardConfig()
    cur_budget = min(churn_budgets, key=lambda b: abs(b - cur.churn_budget))
    rho_current = rho_for(cur.alpha, cur.beta, cur.gamma, cur.churn_coef,
                          cur_budget, cur.c90_coef, cur.probe_coef, cur.q_coef)
    weight_keys = ("alpha", "beta", "gamma", "churn_coef", "churn_budget",
                   "c90_coef", "probe_coef", "q_coef")
    return {
        "recommended": {k: best[k] for k in weight_keys},
        "rho_ranking": best["rho_ranking"],
        "rho_ranking_current_weights": rho_current,
        "top_candidates": candidates[:5],
        "current_weights": {k: getattr(cur, k) for k in weight_keys},
        "forget_weight": float(forget_weight),
        "n_traces": len(traces),
        "trace_policies": [f"{t['policy']}:{t['seed']}" for t in traces],
        "targets": {f"{t['policy']}:{t['seed']}": float(v)
                    for t, v in zip(traces, targets)},
    }


def run_proxy_correlation(
    stream_reader,
    test_readers: List,
    capacity: int,
    n_nearest_neighbours: int = 1,
    patch_shape=None,
    imagesize=None,
    device: Optional[str] = None,
    seed: int = 0,
) -> Dict:
    """Correlate the label-free proxy against labeled AUROC over many banks.

    For every (bank state, stage) pair we compute the proxy coverage/redundancy
    against a held-out normal window of that stage and the labeled image AUROC on
    that stage's test set, then report Spearman correlations. Also fits the
    redundancy weight ``beta`` by a small grid search maximizing |rho|.
    """
    from scipy import stats

    from patchcore.streaming.bank import knn_device

    device = str(device) if device else str(knn_device())
    n_stages = len(test_readers)
    # Per-stage reward scales from that stage's oracle bank (dataset-agnostic).
    scales = {}
    holdouts = {}
    for s in range(n_stages):
        feats = stage_patches(stream_reader, s, max_images=120)
        ref = coreset_bank(feats, capacity, device, seed=1000 + s)
        # held-out window: a fresh sample of that stage's patches
        rng = np.random.default_rng(seed + s)
        hidx = rng.choice(len(feats), size=min(2000, len(feats)), replace=False)
        holdouts[s] = feats[hidx]
        scales[s] = estimate_scales(ref, feats[hidx])

    states = generate_bank_states(stream_reader, capacity, device, seed)

    C_vals, R_vals, auroc_vals = [], [], []
    per_stage = {s: {"C": [], "R": [], "auroc": []} for s in range(n_stages)}
    for bank in states:
        for s in range(n_stages):
            cov_scale, delta = scales[s]
            c = coverage(bank, holdouts[s]) / cov_scale
            r = redundancy(bank, delta)
            m = evaluate_bank_on_stage(
                bank, test_readers[s], n_nearest_neighbours, patch_shape,
                imagesize, device,
            )
            a = m["image_auroc"]
            if not np.isfinite(a):
                continue
            C_vals.append(c); R_vals.append(r); auroc_vals.append(a)
            per_stage[s]["C"].append(c); per_stage[s]["R"].append(r)
            per_stage[s]["auroc"].append(a)

    C = np.asarray(C_vals); R = np.asarray(R_vals); A = np.asarray(auroc_vals)
    rho_C = stats.spearmanr(-C, A).correlation
    rho_R = stats.spearmanr(-R, A).correlation
    best_beta, best_rho = 0.0, rho_C
    for beta in np.linspace(0.0, 2.0, 21):
        rho = stats.spearmanr(-(C + beta * R), A).correlation
        if abs(rho) > abs(best_rho):
            best_rho, best_beta = rho, beta
    per_stage_rho = {
        s: stats.spearmanr(
            -np.asarray(per_stage[s]["C"]), np.asarray(per_stage[s]["auroc"])
        ).correlation
        for s in range(n_stages)
        if len(per_stage[s]["auroc"]) > 2
    }
    return {
        "n_pairs": int(len(A)),
        "rho_coverage": float(rho_C),
        "rho_redundancy": float(rho_R),
        "best_beta": float(best_beta),
        "rho_combined": float(best_rho),
        "per_stage_rho_coverage": {int(k): float(v) for k, v in per_stage_rho.items()},
    }
