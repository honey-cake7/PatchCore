# Streaming RL Memory-Bank Maintenance

Reformulates PatchCore memory-bank maintenance as a budgeted sequential decision
problem under drift. As a stream of *normal* images arrives, a policy under a
fixed budget `M` decides which incoming patch embeddings to admit and which to
evict, trained **label-free** from a proxy reward. See the project plan for
motivation; this file is the operational guide.

## Package map

| module | role |
|---|---|
| `bank.py` | `DynamicMemoryBank` — add/evict/knn under budget; installs into the stock `NearestNeighbourScorer` for identical-to-PatchCore scoring |
| `cache.py` | memmap embedding cache (`EmbeddingCacheWriter/Reader`) |
| `drift.py` | deterministic per-(stage,image) drift transforms + named `DriftSchedule`s |
| `stream.py` | `StreamBuilder` — ordered normal stream + per-stage test manifests |
| `reward.py` | label-free proxy reward: coverage `C`, redundancy `R`, stability `U` |
| `env.py` | `MemoryMaintenanceEnv` — 53-d observation, admission-bar + eviction-utility action, proxy reward |
| `policies.py` | baselines (static/FIFO/random/reservoir/streaming-coreset/periodic-coreset) + `PPOPolicy` |
| `ppo.py` | self-contained PPO (`ActorCritic`, `PPOTrainer`) — no gym/stable-baselines3 |
| `evaluate.py` | per-stage image AUROC / pixel AUROC / PRO |
| `experiments.py` | Gate 1 (headroom) and Gate 2 (proxy validation) |
| `synthetic.py` | drifting-Gaussian fixture (data-free local development) |

Stock files reused unchanged except `metrics.py` (added `compute_pro`).

## Local smoke tests (no data, macOS/CPU)

On macOS, faiss-cpu + torch both link an OpenMP runtime; set these env vars:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1     # macOS only; harmless on cluster
cd patchcore-inspection

python -m pytest test/test_streaming.py -q               # unit tests
python bin/run_gate1.py --synthetic                      # headroom on the fixture
python bin/run_gate2.py --synthetic                      # proxy validation on the fixture
python bin/train_ppo.py --synthetic --total_env_steps 20000 --eval_baselines
python bin/run_streaming_baseline.py --synthetic --ppo_path ppo_policy.pt
```

## Cluster workflow (real data)

1. **Cache embeddings** (only GPU-bound step):

   ```bash
   python bin/cache_embeddings.py \
       --backbone_name wideresnet50 -le layer2 -le layer3 \
       --data_path $DATA/kvasir_patchcore --classname kvasir_patchcore \
       --drift staged_abrupt_4 --seed 0 --out_dir cache/kvasir/wrn50/staged_abrupt_4
   ```

2. **Gate 1 / Gate 2** on the cache (both must pass before policy learning):

   ```bash
   python bin/run_gate1.py --cache_dir cache/kvasir/wrn50/staged_abrupt_4 --capacity 2000
   python bin/run_gate2.py --cache_dir cache/kvasir/wrn50/staged_abrupt_4 --capacity 2000
   ```

3. **Train PPO** on cached streams (train seeds disjoint from eval seeds):

   ```bash
   python bin/train_ppo.py --cache_dir cache/kvasir/wrn50/staged_abrupt_4 \
       --capacity 2000 --total_env_steps 2000000 --out ppo_kvasir.pt
   ```

4. **Benchmark** all policies:

   ```bash
   python bin/run_streaming_baseline.py --cache_dir cache/kvasir/wrn50/staged_abrupt_4 \
       --capacity 2000 --ppo_path ppo_kvasir.pt --seeds 0,1,2 --out results/kvasir
   ```

For the metadata-derived real-drift track, cache with `--drift_mode real` (orders
the stream by `video_id`/anatomical-class parsed from filenames).

## Notes

* `PeriodicCoresetPolicy` mutates the bank in bulk via `replace_bank`, so its
  `n_admit`/`n_evict` churn counters read 0 — compare its cost by index rebuilds,
  not per-vector churn.
* Reward weights `alpha/beta/gamma` are set in `RewardConfig`; Gate 2 fits the
  redundancy weight `beta` by maximizing the proxy↔AUROC Spearman correlation.
