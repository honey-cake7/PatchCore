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
    features: np.ndarray, capacity: int, device="cpu", seed: int = 0
) -> DynamicMemoryBank:
    """Greedy-coreset ``features`` down to ``capacity`` and load into a bank."""
    import torch

    features = np.ascontiguousarray(features, dtype=np.float32)
    if len(features) > capacity:
        pct = capacity / len(features)
        sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(
            percentage=pct, device=torch.device(device)
        )
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
    device: str = "cpu",
    max_images_per_stage: Optional[int] = None,
) -> List[Dict]:
    """Static stage-0 bank vs per-stage oracle coreset, evaluated per stage.

    Pass condition (checked by the caller): static AUROC/PRO drops materially at
    later stages while the oracle holds — that gap is the headroom the learned
    policy can recover.
    """
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
    stream_reader, capacity: int, device: str = "cpu", seed: int = 0
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


def run_proxy_correlation(
    stream_reader,
    test_readers: List,
    capacity: int,
    n_nearest_neighbours: int = 1,
    patch_shape=None,
    imagesize=None,
    device: str = "cpu",
    seed: int = 0,
) -> Dict:
    """Correlate the label-free proxy against labeled AUROC over many banks.

    For every (bank state, stage) pair we compute the proxy coverage/redundancy
    against a held-out normal window of that stage and the labeled image AUROC on
    that stage's test set, then report Spearman correlations. Also fits the
    redundancy weight ``beta`` by a small grid search maximizing |rho|.
    """
    from scipy import stats

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
