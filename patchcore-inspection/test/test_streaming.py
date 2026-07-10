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
