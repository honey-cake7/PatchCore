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
