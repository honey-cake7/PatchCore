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
    stage0_only: bool = False,
) -> List[Dict]:
    """Drive each baseline policy over the stream, recording per-step reward
    components alongside labeled per-stage AUROC and final forgetting.

    One trace per (policy, seed). Because the proxy reward is linear in its
    components, any candidate weighting can later be scored on these traces
    offline (``fit_reward_weights``) — no policy re-runs per candidate.

    ``stage0_only=True`` (opt-in, default off) skips the labeled evaluation
    for drifted stages (1+) and the final stage-0 forgetting re-eval — the
    paper reports stage-0 AUROC only. The stream itself is NOT shortened: the
    policy still runs the full episode and still transitions through every
    stage; only the (expensive) per-stage test-set scoring is pruned. Traces
    produced this way have no ``forget_auroc`` key and ``stage_aurocs``
    containing only stage 0 — ``fit_reward_weights`` raises if such traces are
    combined with a non-zero ``forget_weight``/``drifted_weight``.
    """
    from patchcore.streaming import policies as P
    from patchcore.streaming.env import MemoryMaintenanceEnv

    policy_names = policy_names or list(P.BASELINES)
    seeds = seeds if seeds is not None else [0]

    def eval_fn(env, stage):
        if stage0_only and stage != 0:
            return {}
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
            forget_m = None
            if not stage0_only:
                forget_m = evaluate_bank_on_stage(
                    env.bank, test_readers[0], n_nearest_neighbours, patch_shape,
                    imagesize,
                )
            traces.append(_trace_from_summary(name, seed, summ, forget_m))
    return traces


def _trace_from_summary(
    name: str, seed: int, summ: Dict, forget_m: Optional[Dict]
) -> Dict:
    infos = summ["infos"]
    trace = {
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
        # entries without "image_auroc" are stage0_only skips (evaluate_bank_on_stage
        # never ran for that stage transition) — excluded rather than KeyError'd.
        "stage_aurocs": {
            int(ev["stage"]): float(ev["image_auroc"])
            for ev in summ["evals"] if "image_auroc" in ev
        },
    }
    if forget_m is not None:
        trace["forget_auroc"] = float(forget_m["image_auroc"])
    forget_str = f"{trace['forget_auroc']:.3f}" if forget_m is not None else "skipped(stage0_only)"
    print(f"[trace] {name:26s} seed={seed} "
          f"aurocs={[round(v, 3) for _, v in sorted(trace['stage_aurocs'].items())]} "
          f"forget={forget_str}")
    return trace


def record_ppo_traces(
    stream_reader,
    test_readers: List,
    capacity: int,
    ppo_path: str,
    warmup: int = 100,
    n_nearest_neighbours: int = 1,
    patch_shape=None,
    imagesize=None,
    seeds: Optional[List[int]] = None,
    name: str = "ppo_r1",
    stage0_only: bool = False,
) -> List[Dict]:
    """Replay a trained PPO checkpoint over the stream, recording the same
    component traces as :func:`record_policy_traces`.

    This is the data-collection half of iterated reward refitting: the reward
    is fitted on the heuristic baselines' behavior, and a trained policy can
    maximize the proxy in directions that fitting set never visited while its
    AUROC drops (reward hacking). Adding the trained policy's own trace points
    to the fit exposes those directions to the next round's weight search.

    ``stage0_only`` — see :func:`record_policy_traces`; same opt-in pruning
    of drifted-stage evaluation and the forgetting re-eval.
    """
    from patchcore.streaming import policies as P
    from patchcore.streaming.env import ActionConfig, MemoryMaintenanceEnv
    from patchcore.streaming.ppo import load_checkpoint

    seeds = seeds if seeds is not None else [0]
    ac, norm, meta = load_checkpoint(ppo_path, device="cpu")
    action_mode = (meta or {}).get("action_mode", "continuous6")

    def eval_fn(env, stage):
        if stage0_only and stage != 0:
            return {}
        return evaluate_bank_on_stage(
            env.bank, test_readers[stage], n_nearest_neighbours, patch_shape,
            imagesize,
        )

    traces = []
    for seed in seeds:
        # obs_norm is frozen in v2 checkpoints, so sharing it across seeds is
        # safe; the policy must see the normalization it trained under.
        env = MemoryMaintenanceEnv(
            stream_reader, capacity=capacity, warmup_images=warmup,
            seed=seed, n_nearest_neighbours=n_nearest_neighbours,
            obs_norm=norm, action_cfg=ActionConfig(mode=action_mode),
        )
        summ = P.run_policy(env, P.PPOPolicy(ac), per_stage_eval=eval_fn)
        forget_m = None
        if not stage0_only:
            forget_m = evaluate_bank_on_stage(
                env.bank, test_readers[0], n_nearest_neighbours, patch_shape,
                imagesize,
            )
        traces.append(_trace_from_summary(name, seed, summ, forget_m))
    return traces


def _component_means(traces: List[Dict]) -> Dict[str, np.ndarray]:
    """Per-trace mean of each stored reward component (+ the derived Q)."""
    out = {k: [] for k in ("c", "r", "sd", "c90", "p", "q")}
    for tr in traces:
        out["c"].append(tr["C"].mean())
        out["r"].append(tr["R"].mean())
        out["sd"].append(tr["score_drift"].mean())
        out["c90"].append(tr["C90"].mean() if "C90" in tr else 0.0)
        out["p"].append(tr["P"].mean() if "P" in tr else 0.0)
        out["q"].append(float(np.mean(tr["C90"] / np.maximum(tr["C"], 1e-8)))
                        if "C90" in tr else 0.0)
    return {k: np.asarray(v) for k, v in out.items()}


def permutation_null(
    traces: List[Dict],
    targets: np.ndarray,
    n_permutations: int = 200,
    seed: int = 0,
    alphas=(0.0, 0.25, 1.0),
    betas=(0.0, 0.15, 0.3, 0.6, 1.0),
    gammas=(0.0, 0.1, 0.3),
    churn_coefs=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
    churn_budgets=(0.0, 0.005, 0.01, 0.02),
    c90_coefs=(0.0, 0.25, 0.5, 1.0, 2.0),
    probe_coefs=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
    q_coefs=(0.0, 1.0, 2.0, 4.0, 8.0),
) -> Dict:
    """How high a fitted rho does this grid reach on *shuffled* targets?

    ``fit_reward_weights`` maximizes Spearman rho over ~162k candidates against
    ~12 trace points with no held-out split, so the reported rho is in-sample:
    a large grid can rank noise. This refits the same grid against permuted
    targets to get the null distribution of the *best* rho. The observed fit is
    only evidence of real structure if it clears that null.

    Grids must mirror ``fit_reward_weights``'s defaults, or the null describes
    a different search than the one that produced the number being tested.
    """
    rng = np.random.default_rng(seed)
    m = _component_means(traces)
    excess = {b: np.asarray([np.maximum(tr["churn"] - b, 0.0).mean()
                             for tr in traces]) for b in churn_budgets}

    # every candidate's per-trace proxy, as one [n_candidates, n_traces] matrix
    rows = []
    for budget in churn_budgets:
        ex = excess[budget]
        for alpha in alphas:
            for beta in betas:
                for gamma in gammas:
                    base = alpha * m["c"] + beta * m["r"] + gamma * m["sd"]
                    for coef in churn_coefs:
                        b2 = base + coef * ex
                        for c90c in c90_coefs:
                            b3 = b2 + c90c * m["c90"]
                            for probec in probe_coefs:
                                b4 = b3 + probec * m["p"]
                                for qc in q_coefs:
                                    rows.append(-(b4 + qc * m["q"]))
    P = np.asarray(rows)

    # Spearman == Pearson on ranks, but only with TIE-AVERAGED ranks — targets
    # routinely tie (a class whose stage-0 AUROC saturates at 1.0 for every
    # policy; seeds of one policy scoring identically), and ordinal ranking
    # would silently disagree with scipy's spearmanr.
    from scipy import stats

    try:
        ranks = np.asarray(stats.rankdata(P, axis=1), dtype=np.float64)
    except TypeError:  # scipy < 1.10: rankdata has no axis argument
        ranks = np.vstack([stats.rankdata(row) for row in P])
    ranks -= ranks.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(ranks, axis=1)
    keep = norms > 1e-12
    ranks, norms = ranks[keep], norms[keep]

    def best_rho(t):
        # signed max, matching fit_reward_weights: it selects the highest rho,
        # not the largest |rho| (a strongly anti-correlated candidate is never
        # chosen). Using abs here would inflate both the observed value and the
        # null, and would stop `observed` from reproducing the fitted rho.
        tr_ = np.asarray(stats.rankdata(t), dtype=np.float64)
        tr_ -= tr_.mean()
        tn = np.linalg.norm(tr_)
        if tn < 1e-12:
            return float("nan")
        return float(np.max((ranks @ tr_) / (norms * tn)))

    observed = best_rho(np.asarray(targets, dtype=np.float64))
    null = np.asarray([best_rho(rng.permutation(targets))
                       for _ in range(n_permutations)])
    return {
        "observed_best_rho": observed,
        "null_mean": float(np.nanmean(null)),
        "null_p95": float(np.nanpercentile(null, 95)),
        "null_max": float(np.nanmax(null)),
        "p_value": float(np.mean(null >= observed)),
        "n_candidates": int(keep.sum()),
        "n_traces": len(traces),
        "n_permutations": int(n_permutations),
    }


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
    drifted_weight: float = 1.0,
    stage0_weight: float = 0.0,
) -> Dict:
    """Grid-fit reward weights so mean episode reward ranks policies like AUROC.

    Target per trace is a weighted sum of three labeled quantities:
    ``drifted_weight`` * mean drifted-stage (stage>=1) image AUROC,
    ``forget_weight`` * the stage-0 forgetting AUROC, and ``stage0_weight`` *
    the stage-0 image AUROC. The defaults reproduce the original target
    (drifted + 0.5*forgetting); ``stage0_weight`` exists because that target
    excludes stage 0 entirely, so weights fitted under it optimize something
    the stage-0-only comparison against stock PatchCore never measures. Stage-0
    AUROC is already recorded in every trace, so re-targeting is a seconds-long
    offline refit (``--traces_in``) — no re-recording.

    Reports the Spearman rho of the best candidate and of the current
    production ``RewardConfig`` defaults (the misalignment baseline), plus the
    top-5 candidates so a flat optimum (many near-ties) is visible.

    Traces recorded under ``stage0_only=True`` (see :func:`record_policy_traces`)
    have no ``forget_auroc`` and no drifted-stage (stage>=1) entries in
    ``stage_aurocs``. Combining such traces with a non-zero ``forget_weight``
    or ``drifted_weight`` would silently target 0.0 for those terms rather
    than the intended quantity, so both are checked up front and raise.
    """
    from scipy import stats

    from patchcore.streaming.reward import RewardConfig

    if forget_weight:
        missing_forget = [
            f"{t['policy']}:{t['seed']}" for t in traces if "forget_auroc" not in t
        ]
        if missing_forget:
            raise ValueError(
                f"forget_weight={forget_weight} but {len(missing_forget)}/"
                f"{len(traces)} traces have no forget_auroc — these look like "
                "stage0_only traces (that mode skips the forgetting re-eval): "
                f"{', '.join(missing_forget[:4])}"
                f"{' ...' if len(missing_forget) > 4 else ''}. "
                "Pass forget_weight=0 for stage0-only traces, or re-record "
                "traces without stage0_only."
            )
    if drifted_weight and not any(
        any(s >= 1 for s in tr["stage_aurocs"]) for tr in traces
    ):
        raise ValueError(
            f"drifted_weight={drifted_weight} but no trace has any drifted-stage "
            "(stage>=1) AUROC — these look like stage0_only traces (that mode "
            "skips evaluation for stages 1+). Pass drifted_weight=0 for "
            "stage0-only traces, or re-record traces without stage0_only."
        )

    targets = []
    no_stage0 = []
    mean_c, mean_r, mean_sd, mean_c90, mean_p, mean_q = [], [], [], [], [], []
    for tr in traces:
        drifted = [v for s, v in tr["stage_aurocs"].items() if s >= 1]
        target = 0.0
        if drifted:
            target += drifted_weight * float(np.mean(drifted))
        if forget_weight:
            target += forget_weight * float(tr["forget_auroc"])
        if stage0_weight:
            # streams whose warmup swallows stage 0 (e.g. toothbrush: 60 images,
            # 4 stages) have no stage-0 eval — drop the term for them rather
            # than KeyError, but say so: their fit is not stage-0-targeted.
            stage0 = tr["stage_aurocs"].get(0)
            if stage0 is None:
                no_stage0.append(f"{tr['policy']}:{tr['seed']}")
            else:
                target += stage0_weight * float(stage0)
        targets.append(target)
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
    if no_stage0:
        print(f"[fit] WARNING: stage0_weight={stage0_weight} but "
              f"{len(no_stage0)}/{len(traces)} traces have no stage-0 eval "
              f"({', '.join(no_stage0[:4])}{' ...' if len(no_stage0) > 4 else ''})"
              " — target excludes the stage-0 term for those")
    targets = np.asarray(targets)
    # A constant target makes every Spearman nan, every candidate gets filtered,
    # and the "best" lookup dies with IndexError — leaving no JSON behind, which
    # a caller that ignores the exit code will silently paper over by loading a
    # STALE reward_weights.json from an earlier run. Fail with the cause named.
    if len(targets) and float(np.ptp(targets)) < 1e-12:
        raise ValueError(
            f"ranking target is constant ({targets[0]:.4f} for all "
            f"{len(targets)} traces) — nothing to rank. "
            + ("Every trace lacks a stage-0 eval, so stage0_weight contributes "
               "nothing: this class's warmup consumed stage 0 (e.g. toothbrush). "
               "Give it a non-stage-0 target (drifted_weight/forget_weight) or "
               "lower WARMUP so a stage-0 eval exists."
               if no_stage0 else
               "Check the drifted/forget/stage0 weights and the traces' "
               "stage_aurocs.")
        )
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
        "drifted_weight": float(drifted_weight),
        "stage0_weight": float(stage0_weight),
        "traces_without_stage0": no_stage0,
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
