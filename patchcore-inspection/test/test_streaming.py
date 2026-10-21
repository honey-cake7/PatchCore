"""Unit tests for the streaming / drift-aware memory-bank maintenance stack.

All tests run on synthetic embeddings or hand-built arrays — no dataset or GPU
required — so they are the local correctness gate before cluster runs.
"""
import numpy as np
import pytest

from patchcore import metrics
from patchcore.streaming.bank import DynamicMemoryBank
from patchcore.streaming.cache import EmbeddingCacheReader, EmbeddingCacheWriter
from patchcore.streaming.drift import SCHEDULES
from patchcore.streaming.env import OBS_DIM, MemoryMaintenanceEnv
from patchcore.streaming.evaluate import evaluate_bank_on_stage
from patchcore.streaming import policies as P
from patchcore.streaming.reward import coverage, redundancy
from patchcore.streaming.synthetic import (
    SyntheticConfig, make_all_test_stages, make_synthetic_stream,
)


# ---- DynamicMemoryBank ---------------------------------------------------
def test_bank_knn_matches_bruteforce():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 32)).astype("float32")
    bank = DynamicMemoryBank(capacity=1000, dim=32)
    bank.add(X)
    q = rng.normal(size=(25, 32)).astype("float32")
    d, slots = bank.knn(q, k=1)
    bf = np.sqrt(((q[:, None] - X[None]) ** 2).sum(-1)).min(1)
    assert np.allclose(d[:, 0], bf, atol=1e-3)
    # returned slot ids point at the true nearest vector
    for i in range(len(q)):
        assert np.allclose(bank._store[slots[i, 0]], X[np.argmin(
            ((q[i] - X) ** 2).sum(1))], atol=1e-4)


def test_bank_add_evict_capacity_and_reuse():
    bank = DynamicMemoryBank(capacity=10, dim=4)
    slots = bank.add(np.ones((10, 4), dtype="float32"))
    assert len(bank) == 10 and bank.occupancy == 1.0
    with pytest.raises(ValueError):
        bank.add(np.ones((1, 4), dtype="float32"))  # over capacity
    bank.evict(slots[:3])
    assert len(bank) == 7
    reused = bank.add(np.zeros((3, 4), dtype="float32"))  # reuses freed slots
    assert len(bank) == 10 and set(reused).issubset(set(slots[:3]))


def test_bank_projected_mean_matches_bruteforce_and_caches_device_proj():
    import torch

    rng = np.random.default_rng(7)
    X = rng.normal(size=(50, 8)).astype("float32")
    proj = rng.normal(size=(8, 3)).astype("float32")
    bank = DynamicMemoryBank(capacity=100, dim=8)
    bank.add(X)
    bf = (X @ proj).mean(axis=0)

    out1 = bank.projected_mean(proj)
    assert np.allclose(out1, bf, atol=1e-4)
    # cached device tensor is reused (same numpy proj object) and stays correct
    out2 = bank.projected_mean(proj)
    assert np.allclose(out2, bf, atol=1e-4)
    cache = getattr(bank, "_proj_dev_cache", None)
    assert cache is not None and cache[0] is proj

    # a different numpy array invalidates the cache and still matches
    proj2 = rng.normal(size=(8, 3)).astype("float32")
    bf2 = (X @ proj2).mean(axis=0)
    out3 = bank.projected_mean(proj2)
    assert np.allclose(out3, bf2, atol=1e-4)

    # torch-tensor input (already on device) matches too, mirroring knn()
    out4 = bank.projected_mean(torch.from_numpy(proj))
    assert np.allclose(out4, bf, atol=1e-4)


def test_bank_incremental_mirror_matches_bruteforce():
    """Interleaved add/evict/restore must keep the device mirror consistent.

    The mirror is updated incrementally per mutation (not rebuilt), so drift
    between numpy state and device state would only surface under exactly this
    kind of mutation churn.
    """
    import torch

    rng = np.random.default_rng(3)
    bank = DynamicMemoryBank(capacity=60, dim=16)
    bank.add(rng.normal(size=(40, 16)).astype("float32"))
    bank.knn(rng.normal(size=(4, 16)).astype("float32"))  # builds the mirror
    snap = bank.snapshot()
    for _ in range(30):
        active = bank.active_slots()
        bank.evict(rng.choice(active, size=rng.integers(1, 4), replace=False))
        bank.add(rng.normal(size=(rng.integers(1, 4), 16)).astype("float32"))
        q = rng.normal(size=(8, 16)).astype("float32")
        d, slots = bank.knn(q, k=2)
        ref = bank.vectors()
        bf = np.sqrt(((q[:, None] - ref[None]) ** 2).sum(-1))
        bf.sort(axis=1)
        assert np.allclose(d, bf[:, :2], atol=1e-3)
        assert bank._active[slots].all()  # returned ids are active slots
        # tensor queries (the no-upload hot path) agree with numpy queries
        dt, st = bank.knn(torch.from_numpy(q), k=2)
        assert np.allclose(dt, d, atol=1e-5) and np.array_equal(st, slots)
        # member NN-2 against numpy reference
        nn2 = bank._member_nn2()
        dd = np.sqrt(((ref[:, None] - ref[None]) ** 2).sum(-1))
        np.fill_diagonal(dd, np.inf)
        assert np.allclose(np.sort(nn2), np.sort(dd.min(1)), atol=1e-3)
    bank.restore(snap)
    d, _ = bank.knn(snap.store[snap.active][:5], k=1)
    assert np.allclose(d[:, 0], 0.0, atol=1e-2)  # restored members found


def test_lockstep_collect_matches_sequential():
    """Batched cross-env k-NN stepping must reproduce sequential env.step().

    Covers both reward-config regimes (full components; beta/probe/gamma
    zeroed, which changes which k-NNs are batched vs skipped).
    """
    import torch

    from patchcore.streaming.env import MemoryMaintenanceEnv, RunningNorm
    from patchcore.streaming.ppo import PPOConfig, PPOTrainer
    from patchcore.streaming.reward import RewardConfig

    class LockstepReader:
        def __init__(self, n_images=24, patches=100, dim=16, seed=0):
            rng = np.random.default_rng(seed)
            self.data = rng.normal(size=(n_images, patches, dim)).astype("float32")

        @property
        def n_images(self):
            return len(self.data)

        @property
        def dim(self):
            return self.data.shape[2]

        def image_patches(self, i):
            return self.data[i]

        def stage_of(self, i):
            return 0

        def flat_slice(self, ids):
            return self.data[list(ids)].reshape(-1, self.data.shape[2])

    for beta, probe_c, gamma in [(0.3, 0.5, 0.1), (0.0, 0.0, 0.0)]:
        reader = LockstepReader()

        def env_fns():
            fns = []
            for i in range(2):
                def fn(seed=i):
                    return MemoryMaintenanceEnv(
                        reader, capacity=100, warmup_images=8, seed=seed,
                        reward_cfg=RewardConfig(
                            beta=beta, probe_coef=probe_c, gamma=gamma),
                    )
                fns.append(fn)
            return fns

        batches = {}
        for lockstep in (False, True):
            torch.manual_seed(0)
            cfg = PPOConfig(rollout_steps=20, total_env_steps=40)
            # Frozen norm = production regime (norm_mode=episode). An UNfrozen
            # shared RunningNorm is order-sensitive: sequential stepping
            # interleaves per-env updates differently than phased stepping.
            norm = RunningNorm(OBS_DIM)
            norm.freeze()
            trainer = PPOTrainer(env_fns(), cfg, obs_norm=norm, lockstep=lockstep)
            batches[lockstep] = trainer.collect()

        for key in ("obs", "act", "logp", "adv", "ret"):
            np.testing.assert_allclose(
                batches[False][key], batches[True][key], rtol=1e-4, atol=1e-4,
                err_msg=f"lockstep mismatch in '{key}' (beta={beta})")
        assert abs(batches[False]["mean_reward"] - batches[True]["mean_reward"]) < 1e-5


def test_bank_snapshot_restore():
    rng = np.random.default_rng(1)
    bank = DynamicMemoryBank(capacity=50, dim=8)
    bank.add(rng.normal(size=(30, 8)).astype("float32"))
    snap = bank.snapshot()
    bank.evict(bank.active_slots()[:10])
    bank.add(rng.normal(size=(5, 8)).astype("float32"))
    bank.restore(snap)
    assert len(bank) == 30
    assert np.array_equal(bank.vectors(), snap.store[snap.active])


def test_bank_entry_features_shape():
    bank = DynamicMemoryBank(capacity=40, dim=6)
    bank.add(np.random.default_rng(2).normal(size=(25, 6)).astype("float32"))
    phi = bank.entry_features()
    assert phi.shape == (25, 4) and np.isfinite(phi).all()


# ---- cache round-trip ----------------------------------------------------
def test_cache_round_trip(tmp_path):
    d = str(tmp_path / "c")
    w = EmbeddingCacheWriter(d, n_images=5, patches_per_image=3, dim=4,
                             meta={"stage_ids": [0, 0, 1, 1, 2]})
    for i in range(5):
        w.write(i, np.full((3, 4), i, dtype="float32"))
    w.write_labels([0, 0, 1, 1, 0])
    w.finalize()
    r = EmbeddingCacheReader(d)
    assert r.n_images == 5 and r.patches_per_image == 3 and r.dim == 4
    assert np.array_equal(r.image_patches(3), np.full((3, 4), 3))
    assert r.stage_of(2) == 1 and r.labels.tolist() == [0, 0, 1, 1, 0]
    assert r.flat_slice([0, 4]).shape == (6, 4)


def test_cache_image_patches_memoized_and_evicts(tmp_path):
    d = str(tmp_path / "c")
    w = EmbeddingCacheWriter(d, n_images=5, patches_per_image=3, dim=4,
                             meta={"stage_ids": [0, 0, 1, 1, 2]})
    for i in range(5):
        w.write(i, np.full((3, 4), i, dtype="float32"))
    w.write_labels([0, 0, 1, 1, 0])
    w.finalize()
    r = EmbeddingCacheReader(d)

    def unmemoized(i):
        return np.asarray(r.embeddings[i], dtype=np.float32)

    # Repeated and interleaved indices all match the un-memoized ground
    # truth, whether served from a fresh read or a memo hit.
    for i in [0, 1, 0, 2, 1, 0, 3, 2, 0]:
        got = r.image_patches(i)
        assert np.array_equal(got, unmemoized(i))

    # A memo hit returns the exact cached object (no re-read), and the memo
    # never grows past its 2-slot cap.
    first = r.image_patches(4)
    second = r.image_patches(4)
    assert first is second
    memo = r._patches_memo
    assert len(memo) <= 2

    # Eviction: touching a third distinct index drops the oldest slot, but
    # the evicted index still reads back correctly on the next call.
    memo.clear()
    r.image_patches(0)
    r.image_patches(1)
    assert set(memo.keys()) == {0, 1}
    r.image_patches(2)
    assert set(memo.keys()) == {1, 2}
    assert 0 not in memo
    assert np.array_equal(r.image_patches(0), unmemoized(0))


def test_cache_missing_write_raises(tmp_path):
    w = EmbeddingCacheWriter(str(tmp_path / "c"), 2, 2, 2)
    w.write(0, np.zeros((2, 2), "float32"))
    with pytest.raises(RuntimeError):
        w.finalize()


# ---- PRO metric ----------------------------------------------------------
def test_compute_pro_perfect_and_partial():
    gt = np.zeros((2, 12, 12), dtype=np.uint8)
    gt[0, 2:5, 2:5] = 1
    gt[1, 7:9, 7:9] = 1
    assert metrics.compute_pro(gt.astype(float) * 10, gt)["pro"] > 0.98
    seg = np.zeros((2, 12, 12))
    seg[0, 2:5, 2:5] = 5.0  # only region 0 detected
    partial = metrics.compute_pro(seg, gt)["pro"]
    assert 0.3 < partial < 0.75


def test_compute_pro_no_regions():
    seg = np.random.default_rng(0).random((2, 8, 8))
    gt = np.zeros((2, 8, 8), dtype=np.uint8)
    assert metrics.compute_pro(seg, gt)["pro"] == 0.0


# ---- drift transforms ----------------------------------------------------
def test_drift_deterministic_and_severity_increases():
    import PIL.Image

    img = PIL.Image.fromarray(
        (np.random.default_rng(0).random((64, 64, 3)) * 255).astype("uint8")
    )
    sch = SCHEDULES["staged_abrupt_4"]
    base = np.asarray(img).astype(int)
    prev = -1
    for stage in range(4):
        tf = sch.stage_transform(stage, base_seed=0)
        a = np.asarray(tf(img, "k")).astype(int)
        b = np.asarray(tf(img, "k")).astype(int)
        assert np.array_equal(a, b)  # deterministic per key
        delta = np.abs(a - base).mean()
        assert delta >= prev - 1e-6  # severity monotone across stages
        prev = delta


# ---- reward sign ---------------------------------------------------------
def test_reward_coverage_and_redundancy_signs():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(200, 16)).astype("float32")
    good = DynamicMemoryBank.from_vectors(pts[:150], capacity=150)
    far = DynamicMemoryBank.from_vectors(pts[:150] + 10.0, capacity=150)
    probe = pts[150:]
    assert coverage(good, probe) < coverage(far, probe)  # closer bank covers better
    # a clumped bank is more redundant than a spread one
    spread = DynamicMemoryBank.from_vectors(pts[:150], capacity=150)
    clumped = DynamicMemoryBank.from_vectors(
        np.repeat(pts[:3], 50, axis=0) + rng.normal(scale=1e-3, size=(150, 16)).astype("float32"),
        capacity=150,
    )
    delta = 1.0
    assert redundancy(clumped, delta) > redundancy(spread, delta)


# ---- env -----------------------------------------------------------------
@pytest.fixture(scope="module")
def small_stream():
    cfg = SyntheticConfig(images_per_stage=40, dim=32, patches_per_image=8)
    return make_synthetic_stream(cfg), make_all_test_stages(cfg, n_normal=40, n_anomaly=40)


def _make_env(stream, seed=0):
    return MemoryMaintenanceEnv(stream, capacity=400, warmup_images=40, seed=seed)


def test_env_obs_dim_and_finite(small_stream):
    stream, _ = small_stream
    env = _make_env(stream)
    obs = env.reset()
    assert obs.shape == (OBS_DIM,) and np.isfinite(obs).all()


def test_env_determinism(small_stream):
    stream, _ = small_stream
    actions = np.random.default_rng(3).normal(size=(20, 6))

    def rollout():
        env = _make_env(stream, seed=7)
        env.reset()
        rs = []
        for a in actions:
            _, r, done, _ = env.step(a)
            rs.append(r)
            if done:
                break
        return rs

    assert np.allclose(rollout(), rollout())


def test_env_action_respects_budget(small_stream):
    stream, _ = small_stream
    env = _make_env(stream)
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(30):
        _, _, done, info = env.step(rng.normal(size=6))
        assert len(env.bank) <= env.capacity
        assert info["n_admit"] >= 0 and info["n_evict"] >= 0
        if done:
            break


def test_decode_action_indices_valid(small_stream):
    stream, _ = small_stream
    env = _make_env(stream)
    env.reset()
    admit, evict = env.decode_action(np.random.default_rng(1).normal(size=6))
    assert admit.max(initial=-1) < len(env.current_batch)
    active = set(env.bank.active_slots().tolist())
    assert set(evict.tolist()).issubset(active)


# ---- evaluation + baselines ---------------------------------------------
def test_static_bank_separates_anomalies(small_stream):
    stream, tests = small_stream
    env = _make_env(stream)
    env.reset()
    m = evaluate_bank_on_stage(env.bank, tests[0], n_nearest_neighbours=1)
    assert m["image_auroc"] > 0.8  # stage-0 bank detects stage-0 anomalies


def test_record_policy_traces_stage0_only(small_stream):
    """stage0_only=True must not shorten the episode — only prune which
    stages get labeled evaluation and drop the forgetting re-eval."""
    from patchcore.streaming.experiments import record_policy_traces

    stream, tests = small_stream
    # warmup < images_per_stage (40) so some stage-0 images are stepped
    # through (not swallowed by warmup) and produce a stage-0 boundary eval.
    full = record_policy_traces(
        stream, tests, capacity=400, warmup=35, policy_names=["static"], seeds=[0],
    )
    stage0 = record_policy_traces(
        stream, tests, capacity=400, warmup=35, policy_names=["static"], seeds=[0],
        stage0_only=True,
    )
    assert len(full) == 1 and len(stage0) == 1
    # default (flag off) behaviour: every stage evaluated + forgetting re-eval
    assert "forget_auroc" in full[0]
    assert set(full[0]["stage_aurocs"]) == set(range(len(tests)))
    # stage0_only: forgetting skipped, only stage 0 evaluated
    assert "forget_auroc" not in stage0[0]
    assert set(stage0[0]["stage_aurocs"]) == {0}
    assert np.isfinite(stage0[0]["stage_aurocs"][0])
    # same number of env steps either way -> the stream is NOT truncated
    assert len(full[0]["C"]) == len(stage0[0]["C"])
    assert np.array_equal(full[0]["stage"], stage0[0]["stage"])


def test_adaptive_beats_static_under_drift(small_stream):
    stream, tests = small_stream

    def final_auroc(policy):
        env = _make_env(stream, seed=0)

        def ev(e, s):
            return evaluate_bank_on_stage(e.bank, tests[s], 1)
        summ = P.run_policy(env, policy, per_stage_eval=ev)
        drifted = [x["image_auroc"] for x in summ["evals"] if x["stage"] >= 2]
        return np.mean(drifted)

    static = final_auroc(P.StaticPolicy())
    reservoir = final_auroc(P.ReservoirPolicy())
    assert reservoir >= static  # maintenance helps (or ties) at drifted stages


# ---- PPO -----------------------------------------------------------------
def test_ppo_collect_update_runs(small_stream):
    stream, _ = small_stream
    from patchcore.streaming.ppo import PPOConfig, PPOTrainer

    trainer = PPOTrainer(
        [lambda: _make_env(stream, seed=i) for i in range(2)],
        PPOConfig(rollout_steps=16, minibatches=2, epochs=2, total_env_steps=64),
    )
    batch = trainer.collect()
    assert batch["obs"].shape[1] == OBS_DIM
    assert np.isfinite(batch["ret"]).all()
    trainer.update(batch)  # should not raise


# ---- reward-weight fit target --------------------------------------------
def _fake_traces():
    """Two traces with hand-set component means and labeled AUROCs."""
    def tr(policy, c, stage_aurocs, forget):
        n = 8
        return {
            "policy": policy, "seed": 0,
            "C": np.full(n, c), "R": np.full(n, 0.1),
            "churn": np.full(n, 0.02), "score_drift": np.full(n, 0.01),
            "C90": np.full(n, 2 * c), "P": np.full(n, c),
            "stage": np.zeros(n, dtype=np.int64),
            "stage_aurocs": stage_aurocs, "forget_auroc": forget,
        }
    return [
        tr("a", 0.5, {0: 0.90, 1: 0.80, 2: 0.70}, 0.60),
        tr("b", 0.4, {0: 0.70, 1: 0.85, 2: 0.75}, 0.95),
    ]


def _targets_of(result, traces):
    return [result["targets"][f"{t['policy']}:{t['seed']}"] for t in traces]


def test_fit_target_defaults_match_historical_formula():
    from patchcore.streaming.experiments import fit_reward_weights

    traces = _fake_traces()
    res = fit_reward_weights(traces)
    expected = [np.mean([0.80, 0.70]) + 0.5 * 0.60,
                np.mean([0.85, 0.75]) + 0.5 * 0.95]
    assert _targets_of(res, traces) == pytest.approx(expected)
    assert res["drifted_weight"] == 1.0 and res["stage0_weight"] == 0.0


def test_fit_target_stage0_only():
    from patchcore.streaming.experiments import fit_reward_weights

    traces = _fake_traces()
    res = fit_reward_weights(traces, drifted_weight=0.0, forget_weight=0.0,
                             stage0_weight=1.0)
    assert _targets_of(res, traces) == pytest.approx([0.90, 0.70])
    # stage 0 ranks the traces opposite to the default target -> the fitted
    # weights must differ; this is the whole point of the knob
    assert res["stage0_weight"] == 1.0
    assert not res["traces_without_stage0"]


def test_fit_target_handles_missing_stage0():
    from patchcore.streaming.experiments import fit_reward_weights

    traces = _fake_traces()
    del traces[0]["stage_aurocs"][0]  # toothbrush case: warmup swallowed stage 0
    res = fit_reward_weights(traces, drifted_weight=0.0, forget_weight=0.0,
                             stage0_weight=1.0)
    assert _targets_of(res, traces) == pytest.approx([0.0, 0.70])
    assert res["traces_without_stage0"] == ["a:0"]


def _fake_stage0_only_traces():
    """Traces shaped like record_policy_traces(..., stage0_only=True) output:
    no forget_auroc key, and stage_aurocs holds only the stage-0 entry."""
    traces = _fake_traces()
    for tr in traces:
        del tr["forget_auroc"]
        tr["stage_aurocs"] = {0: tr["stage_aurocs"][0]}
    return traces


def test_fit_stage0_only_traces_reject_nonzero_forget_weight():
    from patchcore.streaming.experiments import fit_reward_weights

    traces = _fake_stage0_only_traces()
    # forget_weight defaults to 0.5 (the historical target) — stage0_only
    # traces have no forget_auroc, so this must fail loudly rather than
    # silently dropping the term (and quietly fitting a wrong target).
    with pytest.raises(ValueError, match="forget_weight"):
        fit_reward_weights(traces, drifted_weight=0.0)


def test_fit_stage0_only_traces_reject_nonzero_drifted_weight():
    from patchcore.streaming.experiments import fit_reward_weights

    traces = _fake_stage0_only_traces()
    # drifted_weight defaults to 1.0 — stage0_only traces have no stage>=1
    # entries in stage_aurocs, so this must fail loudly too.
    with pytest.raises(ValueError, match="drifted_weight"):
        fit_reward_weights(traces, forget_weight=0.0)


def test_fit_stage0_only_traces_ok_with_stage0_weight():
    from patchcore.streaming.experiments import fit_reward_weights

    traces = _fake_stage0_only_traces()
    # the intended stage0_only pairing: zero out drifted/forget, target stage0
    res = fit_reward_weights(traces, drifted_weight=0.0, forget_weight=0.0,
                             stage0_weight=1.0)
    assert _targets_of(res, traces) == pytest.approx([0.90, 0.70])
    assert not res["traces_without_stage0"]


# ---- reviewer protocol: calibration split + pre-shift episodes -----------
def test_calibration_split_disjoint_stratified_and_preshift_stops_at_stage1():
    from patchcore.streaming.cache import calibration_split, stage0_end

    labels = np.array([0] * 30 + [1] * 70)
    calib, eval_ = calibration_split(labels, 0.3, seed=0)
    assert not set(calib) & set(eval_)                    # disjoint
    assert len(calib) + len(eval_) == len(labels)         # covers every image
    assert (labels[calib] == 0).sum() == 9 and (labels[calib] == 1).sum() == 21

    cfg = SyntheticConfig(images_per_stage=20, patches_per_image=64)
    stream = make_synthetic_stream(cfg)
    stop = stage0_end(stream)
    assert stop > 10 and stream.stage_of(stop) == 1 and stream.stage_of(stop - 1) == 0
    env = MemoryMaintenanceEnv(stream, capacity=128, warmup_images=10, episode_stop=stop)
    summ = P.run_policy(env, P.FIFOPolicy(k=8))
    assert len(summ["infos"]) == stop - 10                # ends at the first stage-1 image
    assert all(i["stage"] == 0 for i in summ["infos"])


# ==== Agent B block: kcenter / myopic_iota / benchmark CSV / full memory ====
def _load_bin(name):
    """Import a bin/ script as a module (bin/ is not a package)."""
    import importlib.util
    import os

    path = os.path.join(os.path.dirname(__file__), "..", "bin", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_bin_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_greedy_net_matches_sequential_rule():
    import torch

    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.normal(size=(300, 4)).astype("float32"))
    for r in (0.3, 1.0, 2.5):
        keep = P.greedy_net(x, r).numpy()
        d = torch.cdist(x, x).numpy()
        ref = []
        for i in range(len(x)):  # the sequential definition of a greedy r-net
            if all(d[i, j] > r for j in ref):
                ref.append(i)
        want = np.zeros(len(x), dtype=bool)
        want[ref] = True
        assert np.array_equal(keep, want), r


def test_kcenter_respects_capacity_and_keeps_r_net(small_stream):
    stream, _ = small_stream
    env = MemoryMaintenanceEnv(stream, capacity=150, warmup_images=40, seed=0)
    pol = P.BASELINES["kcenter"]()
    env.reset()
    done, doublings, r0 = False, 0, None
    while not done:
        r_before = pol.r
        _, _, done, _ = pol.step(env)
        r0 = r0 if r0 is not None else r_before
        assert len(env.bank) <= env.capacity
    assert pol.r > r0  # the stream drifted -> at least one doubling happened
    v = env.bank.vectors()
    dd = np.sqrt(((v[:, None] - v[None]) ** 2).sum(-1))
    np.fill_diagonal(dd, np.inf)
    assert dd.min() > pol.r - 1e-4  # bank stays r-separated


def test_trial_mutation_restores_bank_exactly():
    import torch

    rng = np.random.default_rng(5)
    bank = DynamicMemoryBank(capacity=50, dim=8)
    bank.add(rng.normal(size=(45, 8)).astype("float32"))
    bank.evict(np.array([3, 7]))           # holes below the tail free slots
    q = rng.normal(size=(20, 8)).astype("float32")
    d0, s0 = bank.knn(q, k=3)
    snap = bank.snapshot()
    new = rng.normal(size=(9, 8)).astype("float32")
    with P.trial_mutation(bank, np.array([1, 2, 10, 11]), new) as b:
        assert len(b) == 43 - 4 + 9
        d1, _ = b.knn(q, k=3)
        ref = b.vectors()
        bf = np.sort(np.sqrt(((q[:, None] - ref[None]) ** 2).sum(-1)), axis=1)[:, :3]
        assert np.allclose(d1, bf, atol=1e-3)  # trial state is live on device
    for a in ("store", "active", "insert_step", "insert_stage", "hit_count",
              "last_hit_step"):
        got = bank._store if a == "store" else getattr(bank, "_active" if a == "active" else a)
        assert np.array_equal(got, getattr(snap, a)), a
    assert len(bank) == snap.size and bank.step == snap.step
    assert torch.equal(bank._active_dev.cpu(), torch.from_numpy(snap.active))
    assert torch.allclose(bank._base.cpu()[snap.active],
                          torch.from_numpy(snap.store[snap.active]))
    d2, s2 = bank.knn(q, k=3)
    assert np.array_equal(s2, s0) and np.allclose(d2, d0)


def test_full_memory_chunked_eval_matches_unchunked(small_stream):
    from patchcore.streaming.evaluate import _torch_patch_scores

    off = _load_bin("run_offline_coreset")
    rng = np.random.default_rng(1)
    bank = rng.normal(size=(1000, 16)).astype("float32")
    q = rng.normal(size=(301, 16)).astype("float32")
    ref = _torch_patch_scores(DynamicMemoryBank.from_vectors(bank, 1000), q, 3)
    got = off.chunked_patch_scores(bank, q, 3, bank_chunk=97, query_chunk=50)
    assert np.allclose(got, ref, rtol=1e-4, atol=1e-4)
    # tiny final bank chunk (< k rows) must still merge exactly
    got = off.chunked_patch_scores(bank[:202], q, 3, bank_chunk=100, query_chunk=64)
    ref = _torch_patch_scores(DynamicMemoryBank.from_vectors(bank[:202], 202), q, 3)
    assert np.allclose(got, ref, rtol=1e-4, atol=1e-4)

    stream, tests = small_stream
    feats = stream.flat_slice(range(40))
    m_full = off.evaluate_full_memory(feats, tests[0], 1, None, None,
                                      bank_chunk=33, query_chunk=17)
    m_ref = evaluate_bank_on_stage(
        DynamicMemoryBank.from_vectors(feats, len(feats)), tests[0], 1)
    assert m_full["image_auroc"] == pytest.approx(m_ref["image_auroc"], abs=1e-6)


def test_ppo_path_parsing():
    rsb = _load_bin("run_streaming_baseline")
    assert rsb._parse_ppo_paths("ppo_policy.pt") == [("", "ppo_policy.pt")]
    assert rsb._parse_ppo_paths("0:a/p0.pt, 1:b/p1.pt") == [("0", "a/p0.pt"),
                                                           ("1", "b/p1.pt")]
    assert rsb._parse_ppo_paths("C:dir/x.pt") == [("", "C:dir/x.pt")]


def test_streaming_baseline_csv_superset(tmp_path):
    import csv

    from click.testing import CliRunner

    rsb = _load_bin("run_streaming_baseline")
    res = CliRunner().invoke(rsb.main, [
        "--synthetic", "--capacity", "300", "--policies", "static,kcenter",
        "--seeds", "0", "--stage0_every_boundary", "--out", str(tmp_path)])
    assert res.exit_code == 0, res.output
    with open(tmp_path / "results.csv") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ["policy", "seed", "train_seed", "stage", "image_auroc",
                             "pixel_auroc", "pro", "writes", "evict", "iota"]
    kc = [r for r in rows if r["policy"] == "kcenter"]
    assert {r["stage"] for r in kc} >= {"0", "3", "stage0_at_1", "final_forgetting"}
    assert len({r["writes"] for r in kc}) == 1 and int(kc[0]["writes"]) > 0
    assert all(r["train_seed"] == "" for r in rows)
    # myopic_iota without a held-out set is refused up front
    res = CliRunner().invoke(rsb.main, [
        "--synthetic", "--policies", "myopic_iota", "--out", str(tmp_path)])
    assert res.exit_code != 0 and "heldout_frac" in res.output


def test_myopic_iota_lookahead(small_stream):
    import inspect

    if "heldout_frac" not in inspect.signature(MemoryMaintenanceEnv).parameters:
        pytest.skip("MemoryMaintenanceEnv has no heldout_frac yet (Agent A's "
                    "ι env not integrated) — run after integration")
    stream, _ = small_stream

    def make():
        return MemoryMaintenanceEnv(stream, capacity=150, warmup_images=40,
                                    seed=0, heldout_frac=0.2, heldout_seed=0)

    env = make()
    pol = P.BASELINES["myopic_iota"]()
    env.reset()
    assert len(env.heldout_ids) > 0
    done = False
    while not done:
        snap = env.bank.snapshot()
        admit, evict = pol.decide(env)
        # the trials must leave the live bank untouched
        assert np.array_equal(env.bank._active, snap.active)
        assert np.array_equal(env.bank._store[snap.active], snap.store[snap.active])
        assert env.bank.step == snap.step
        _, _, done, _ = env.step_with_decision(admit, evict)
        assert len(env.bank) <= env.capacity
    myopic = np.mean(np.log(env.heldout_image_scores()))

    env_s = make()
    P.run_policy(env_s, P.StaticPolicy())
    assert np.array_equal(env_s.heldout_ids, env.heldout_ids)  # same H
    static = np.mean(np.log(env_s.heldout_image_scores()))
    assert myopic < static  # lookahead lowers held-out normal scores


def test_myopic_iota_fast_path_matches_trial_banks(small_stream):
    import inspect

    if "heldout_frac" not in inspect.signature(MemoryMaintenanceEnv).parameters:
        pytest.skip("MemoryMaintenanceEnv has no heldout_frac yet")
    stream, _ = small_stream
    # k=3 and a full 120-slot bank: the m=16/64 candidates evict > k slots
    env = MemoryMaintenanceEnv(stream, capacity=120, warmup_images=40, seed=0,
                               n_nearest_neighbours=3, heldout_frac=0.2,
                               heldout_seed=0)
    fast, slow = P.MyopicIotaPolicy(), P.MyopicIotaPolicy(exact_trials=True)
    env.reset()
    done, max_evict, compared = False, 0, 0
    while not done:
        if len(env.heldout_seen_ids()):
            cands, order = fast.candidates(env)
            got = fast._merged_scores(env, cands, order)
            for (admit, ev), g in zip(cands, got):
                with P.trial_mutation(env.bank, ev, env.current_batch[admit]):
                    ref = env.heldout_image_scores(env.bank)
                np.testing.assert_allclose(g, ref, rtol=1e-5, atol=1e-6)
                max_evict = max(max_evict, len(ev))
            compared += 1
        a_f, e_f = fast.decide(env)
        a_s, e_s = slow.decide(env)
        assert np.array_equal(a_f, a_s) and np.array_equal(e_f, e_s)
        _, _, done, _ = env.step_with_decision(a_f, e_f)
    assert compared > 5 and max_evict > 3  # exercised evictions > k


# ==== Agent A: ι reward (teacher table, held-out set, κ cap, S_B tracker) ====
def _iota_stream():
    # warmup 40 x 8 patches > the 256-patch probe, so the bank is non-empty
    cfg = SyntheticConfig(images_per_stage=25, dim=16, patches_per_image=8)
    return make_synthetic_stream(cfg)


def test_teacher_matches_bruteforce_with_clusters():
    from patchcore.streaming.synthetic import ArrayReader
    from patchcore.streaming.teacher import build_teacher

    rng = np.random.default_rng(0)
    E = rng.normal(size=(12, 5, 3)).astype("float32")
    cl = np.array([0, 1, 1, 2, 3, 4, 5, 6, 6, 7, 8, 9])
    T = build_teacher(ArrayReader(E, np.zeros(12)), 3, clusters=cl, block=5,
                      verbose=False)
    S = np.full((12, 12), np.nan)
    for n in range(12):
        for t in range(n, 12):
            F = [E[m] for m in range(t + 1) if cl[m] != cl[n]]
            if not F:
                continue
            d = ((E[n][:, None] - np.concatenate(F)[None]) ** 2).sum(-1)
            d.sort(axis=1)
            S[n, t] = d[:, :3].mean(axis=1).max()
    assert np.array_equal(np.isnan(S), np.isnan(T["S_F"]))
    assert np.nanmax(np.abs(S - T["S_F"])) < 1e-4
    assert T["eps"] == pytest.approx(0.01 * np.nanmedian(S[:, -1]), rel=1e-5)


def test_iota_nonnegative_heldout_withheld_and_kappa():
    """Singleton clusters => every bank is a subset of the teacher set, so
    ι >= 0 (float noise aside); held-out images admit nothing; κ caps writes.
    Also exercises every baseline on empty admissible batches."""
    from patchcore.streaming.teacher import build_teacher

    stream = _iota_stream()
    teacher = build_teacher(stream, 3, verbose=False)
    pols = [P.StaticPolicy(), P.FIFOPolicy(k=8), P.RandomPolicy(k=8),
            P.ReservoirPolicy(), P.StreamingGreedyCoresetPolicy(),
            P.PeriodicCoresetPolicy(period=7)]
    for pol in pols:
        env = MemoryMaintenanceEnv(
            stream, capacity=60, warmup_images=40, seed=0, reward_mode="iota",
            teacher=teacher, heldout_frac=0.1, heldout_seed=3, admit_cap_abs=4)
        summ = P.run_policy(env, pol)
        H = set(env.heldout_ids.tolist())
        assert len(H) == 6
        iotas = np.array([i["iota"] for i in summ["infos"]])
        seen = np.isfinite(iotas)
        assert seen.any() and (iotas[seen] >= -1e-4).all(), pol.name
        assert np.allclose(summ["rewards"], np.where(seen, -np.nan_to_num(iotas), 0.0))
        for t, info in zip(range(40, stream.n_images), summ["infos"]):
            if t in H:
                assert info["n_admit"] == 0 or pol.name == "periodic_coreset"
            if pol.name != "periodic_coreset":   # replace_bank is κ-exempt
                assert info["writes"] <= 4
    # default PPO decode on a held-out step: empty admissible -> no-op admits
    env = MemoryMaintenanceEnv(stream, capacity=60, warmup_images=40,
                               reward_mode="iota", teacher=teacher, heldout_frac=0.1)
    env.reset()
    while env._t not in set(env.heldout_ids.tolist()):
        env.step(np.zeros(6))
    assert len(env.current_batch) == 0
    admit, evict = env.decode_action(np.ones(6))
    assert len(admit) == 0 and len(evict) == 0


def test_iota_incremental_sb_matches_full_recompute():
    from patchcore.streaming.teacher import build_teacher, image_scores_from_topk

    stream = _iota_stream()
    env = MemoryMaintenanceEnv(
        stream, capacity=50, warmup_images=40, seed=1, reward_mode="iota",
        teacher=build_teacher(stream, 3, verbose=False), heldout_frac=0.4)
    env.reset()
    rng = np.random.default_rng(0)
    checked = 0
    for step in range(60):
        A = len(env.current_batch)
        admit = rng.choice(A, size=rng.integers(0, A + 1), replace=False) if A \
            else np.empty(0, np.int64)
        slots = env.bank.active_slots()
        # mostly small evictions (incremental path), sometimes > 25% churn
        n_ev = rng.integers(0, 4) if step % 9 else rng.integers(15, 30)
        evict = rng.choice(slots, size=min(n_ev, len(slots)), replace=False)
        if step == 25:
            env.replace_bank(env.bank.vectors()[rng.permutation(len(env.bank))[:30]])
        _, _, done, _ = env.step_with_decision(admit, evict)
        n = env._ho_nimg
        if n:
            inc = image_scores_from_topk(env._ho_topd[:n * env._ho_ppi], n).cpu().numpy()
            full = env.heldout_image_scores()[:n]
            np.testing.assert_allclose(inc, full, rtol=1e-5, atol=1e-6)
            checked += 1
        if done:
            break
    assert checked > 20


def test_lockstep_iota_matches_sequential_and_checkpoint_meta(tmp_path):
    import torch

    from patchcore.streaming.env import RunningNorm
    from patchcore.streaming.ppo import PPOConfig, PPOTrainer, load_checkpoint
    from patchcore.streaming.teacher import build_teacher

    stream = _iota_stream()
    teacher = build_teacher(stream, 3, verbose=False)

    def env_fns():
        return [lambda s=i: MemoryMaintenanceEnv(
            stream, capacity=60, warmup_images=40, seed=s, reward_mode="iota",
            teacher=teacher, heldout_frac=0.2, heldout_seed=0, admit_cap_abs=4)
            for i in range(3)]

    batches = {}
    for lockstep in (False, True):
        torch.manual_seed(0)
        cfg = PPOConfig(rollout_steps=40, total_env_steps=120, adv_mode="grpo",
                        adv_baseline="loo", hidden=32, epochs=1, minibatches=4)
        norm = RunningNorm(OBS_DIM)
        norm.freeze()
        trainer = PPOTrainer(env_fns(), cfg, obs_norm=norm, lockstep=lockstep)
        batches[lockstep] = trainer.collect()
    for key in ("obs", "act", "logp", "adv", "ret"):
        np.testing.assert_allclose(batches[False][key], batches[True][key],
                                   rtol=1e-4, atol=1e-4, err_msg=key)
    assert batches[True]["perf"]["iota"] > 0
    trainer.update(batches[True])
    path = str(tmp_path / "p.pt")
    trainer.save(path)
    ac, _, meta = load_checkpoint(path)
    assert meta["reward_mode"] == "iota" and meta["hidden"] == 32
    assert meta["admit_cap_abs"] == 4 and meta["heldout_frac"] == 0.2
    assert ac.mean_head.in_features == 32



# ---- band6 action decoding: admit from a novelty window -------------------
def test_band6_admits_typical_or_novel_window_nearest_centre_first():
    from patchcore.streaming.env import ActionConfig

    stream = make_synthetic_stream(SyntheticConfig(images_per_stage=20, patches_per_image=64))
    env = MemoryMaintenanceEnv(stream, capacity=256, warmup_images=10,
                               action_cfg=ActionConfig(mode="band6", admit_cap_frac=0.05))
    env.reset()
    nn = env.batch_novelty
    ranked = np.argsort(nn, kind="stable")
    cap = int(0.05 * 256)                                   # a[1] = +inf -> full cap
    typ, _ = env.decode_action(np.array([-50., 50, 0, 0, 0, 0]))
    nov, _ = env.decode_action(np.array([50., 50, 0, 0, 0, 0]))
    assert len(typ) == len(nov) == cap
    assert set(typ) == set(ranked[:cap])                    # least novel patches
    assert set(nov) == set(ranked[-cap:])                   # most novel patches
    assert typ[0] == ranked[0] and nov[0] == ranked[-1]     # nearest the centre first
    mid, _ = env.decode_action(np.array([0., 50, 0, 0, 0, 0]))
    r = np.searchsorted(np.sort(nn), nn[mid])
    assert abs(np.median(r) - (len(nn) - 1) / 2) <= cap      # window sits mid-ranking


# ---- Tier-1 policy fixes: no-op, Beta policy, delta ι, KL brake, eval select --
def _tier1_stream():
    return make_synthetic_stream(SyntheticConfig(images_per_stage=12, patches_per_image=64))


def test_admit_floor_zero_allows_exact_noop_for_both_modes():
    from patchcore.streaming.env import ActionConfig

    for mode in ("continuous6", "band6"):
        noop = np.array([0.0, -50, 0, 0, 0, 0])            # a1 -> minimum count
        for floor, expect in ((1, 1), (0, 0)):
            env = MemoryMaintenanceEnv(_tier1_stream(), capacity=256, warmup_images=8,
                                       action_cfg=ActionConfig(mode=mode, admit_floor=floor))
            env.reset()
            admit, evict = env.decode_action(noop)
            assert len(admit) == expect, (mode, floor)
            if floor == 0:
                assert len(evict) == 0


def test_beta_policy_is_bounded_and_round_trips(tmp_path):
    import torch

    from patchcore.streaming.env import ActionConfig
    from patchcore.streaming.ppo import (
        ActorCritic, PPOConfig, PPOTrainer, action_cfg_from_meta, load_checkpoint,
    )

    ac = ActorCritic(hidden=16, dist="beta")
    with torch.no_grad():                                    # push logits to extremes
        ac.conc_head.bias[:6] = 1e4
        ac.conc_head.bias[6:] = -1e4
    with torch.no_grad():
        dist, _ = ac(torch.zeros(4, OBS_DIM))
        a = np.tanh(ac.to_env(ac.mean_action(dist)).numpy())
    assert np.all(np.abs(a) < 1.0) and np.all(np.isfinite(a))   # cannot run off to ±inf
    x = ac.sample(dist)
    assert torch.isfinite(dist.log_prob(x)).all()

    stream = _tier1_stream()
    fn = lambda: MemoryMaintenanceEnv(stream, capacity=128, warmup_images=8,  # noqa: E731
                                      action_cfg=ActionConfig(mode="band6", admit_floor=0))
    tr = PPOTrainer([fn], PPOConfig(rollout_steps=8, total_env_steps=8, hidden=16,
                                    policy_dist="beta"))
    path = tmp_path / "p.pt"
    tr.save(str(path))
    ac2, _, meta = load_checkpoint(str(path))
    assert ac2.dist_kind == "beta" and meta["policy_dist"] == "beta"
    cfg = action_cfg_from_meta(meta)
    assert cfg.mode == "band6" and cfg.admit_floor == 0
    for p1, p2 in zip(tr.ac.state_dict().values(), ac2.state_dict().values()):
        assert torch.equal(p1, p2)


def test_delta_iota_reward_telescopes_to_final_iota():
    from patchcore.streaming.reward import RewardConfig
    from patchcore.streaming.teacher import build_teacher

    stream = _tier1_stream()
    teacher = build_teacher(stream, 5, verbose=False)
    rets = {}
    for form in ("level", "delta"):
        env = MemoryMaintenanceEnv(stream, capacity=128, warmup_images=8, seed=0,
                                   reward_cfg=RewardConfig(reward_form=form),
                                   reward_mode="iota", teacher=teacher,
                                   heldout_frac=0.2, heldout_seed=3)
        summ = P.run_policy(env, P.FIFOPolicy(k=4))
        rets[form] = (summ["rewards"], [i["iota"] for i in summ["infos"]])
    level_r, iotas = rets["level"]
    delta_r, _ = rets["delta"]
    final = [v for v in iotas if np.isfinite(v)][-1]
    assert np.isclose(np.sum(delta_r), -final, atol=1e-5)    # Σ -(ι_t - ι_{t-1}) = -ι_T
    assert np.allclose(level_r, [-(v if np.isfinite(v) else 0.0) for v in iotas])


def test_target_kl_stops_update_early_and_eval_select_keeps_best():
    import torch

    from patchcore.streaming.ppo import PPOConfig, PPOTrainer

    stream = _tier1_stream()
    fns = [lambda s=i: MemoryMaintenanceEnv(stream, capacity=128, warmup_images=8, seed=s)
           for i in range(2)]
    torch.manual_seed(0)
    tr = PPOTrainer(fns, PPOConfig(rollout_steps=16, total_env_steps=32, lr=0.5,
                                   target_kl=1e-6, epochs=4, minibatches=4))
    steps, stopped = tr.update(tr.collect())
    assert stopped and steps < 4 * 4                         # brake fired before the end

    scores = iter([1.0, 5.0, 2.0])
    snaps = []

    def eval_fn(ac, norm):
        snaps.append({k: v.clone() for k, v in ac.state_dict().items()})
        return next(scores)

    tr = PPOTrainer(fns, PPOConfig(rollout_steps=8, total_env_steps=8 * 2 * 3,
                                   select="eval", eval_every=1), eval_fn=eval_fn)
    tr.train(log_every=10)
    for k, v in tr.ac.state_dict().items():                  # restored = the 5.0 snapshot
        assert torch.equal(v, snaps[1][k])
