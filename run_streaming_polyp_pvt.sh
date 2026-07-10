#!/bin/bash
#SBATCH --job-name=streaming-polyp-pvt
#SBATCH --partition=LocalQ
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --gres=shard:32
#SBATCH --ntasks=1
#SBATCH --output=/home/user1/aniket/Patchcore/logs/streaming_polyppvt%j.out
#SBATCH --error=/home/user1/aniket/Patchcore/logs/streaming_polyppvt%j.err
#SBATCH --cpus-per-task=32
# ──────────────────────────────────────────────────────────────────────
# run_streaming_polyp_pvt.sh
# End-to-end RL memory-bank maintenance pipeline with the Polyp-PVT backbone:
#   1. cache patch embeddings over a drift-ordered normal stream + per-stage tests
#   2. Gate 1  — headroom  (does drift hurt a static bank?)
#   3. Gate 2  — proxy validation (does the label-free reward track AUROC?)
#   4. train the self-contained PPO maintenance policy
#   5. benchmark all policies (baselines + PPO), writing per-stage AUROC/PRO
#
# RL is only justified if BOTH gates pass. By default this script STOPS if a gate
# fails; pass FORCE=1 to run every step regardless.
#
# Usage:
#   sbatch run_streaming_polyp_pvt.sh                 # cluster (SLURM)
#   bash   run_streaming_polyp_pvt.sh                 # interactive
#   FORCE=1 DRIFT=staged_gradual_4 bash run_streaming_polyp_pvt.sh
# ──────────────────────────────────────────────────────────────────────
set -uo pipefail

# ─── environment ──────────────────────────────────────────────────────
# Defaults to the cluster setup (this script is meant for `sbatch`). Under
# SLURM `module` is a shell function that is NOT auto-defined, so we must load
# it exactly like train_polyp_pvt.sh rather than probing with `command -v`.
# For a laptop/CPU run pass LOCAL=1 (uses python3 and the repo-relative path).
LOCAL=${LOCAL:-0}
PY=${PY:-python}
REPO_DIR=${REPO_DIR:-/home/user1/aniket/Patchcore/PatchCore/patchcore-inspection}

if [ "${LOCAL}" = "1" ]; then
  cd "$(cd "$(dirname "$0")" && pwd)/patchcore-inspection" || { echo "cannot cd to package dir"; exit 1; }
  PY=${PY:-python3}
else
  cd "${REPO_DIR}" || { echo "cannot cd to ${REPO_DIR} (set REPO_DIR)"; exit 1; }
  module load compilers/anaconda3-2024.06
  module load libs/cuda-12.8
  source /apps/compilers/anaconda3-2024.06/etc/profile.d/conda.sh
  conda activate patchcore
fi

# ─── CONFIG (edit these) ──────────────────────────────────────────────
DATA_PATH=${DATA_PATH:-/home/user1/aniket/Patchcore/dataset/kvasir_patchcore}
CLASSNAME=${CLASSNAME:-kvasir}                 # subfolder under DATA_PATH (mvtec-style)
BACKBONE=${BACKBONE:-polyp-pvt}                # or pvtv2_b2 (ImageNet) to compare
DRIFT=${DRIFT:-staged_abrupt_4}                # staged_abrupt_4 | staged_gradual_4 | staged_cyclic_4
DRIFT_MODE=${DRIFT_MODE:-synthetic}            # synthetic | real (metadata-ordered)
SEED=${SEED:-0}

# Polyp-PVT (PVTv2-B2) settings — mirror train_polyp_pvt.sh:
#   norm2 (stride/8, 128ch) + norm3 (stride/16, 320ch) ≈ resnet layer2+layer3.
LAYERS=(-le norm2 -le norm3)
PATCHSIZE=${PATCHSIZE:-6}
PRE_DIM=${PRE_DIM:-1024}
TGT_DIM=${TGT_DIM:-1024}
RESIZE=${RESIZE:-256}
IMAGESIZE=${IMAGESIZE:-224}

# streaming / RL settings
CAPACITY=${CAPACITY:-2000}                     # memory budget M
WARMUP=${WARMUP:-100}                          # warmup images for stage-0 bank + reward scales
N_NN=${N_NN:-5}                                # k for k-NN scoring (matches train_polyp_pvt.sh)
PPO_STEPS=${PPO_STEPS:-2000000}
TRAIN_SEEDS=${TRAIN_SEEDS:-0}                  # PPO training seed(s)
EVAL_SEEDS=${EVAL_SEEDS:-0,1,2}               # benchmark eval seeds (disjoint from train ideally)
POLICIES=${POLICIES:-static,fifo,reservoir,streaming_greedy_coreset,periodic_coreset,ppo}

# output locations
TAG=${TAG:-${CLASSNAME}_polyppvt_${DRIFT}}
CACHE_DIR=${CACHE_DIR:-cache/${TAG}}
RESULT_DIR=${RESULT_DIR:-results/streaming/${TAG}}
PPO_OUT=${PPO_OUT:-${RESULT_DIR}/ppo_${TAG}.pt}
mkdir -p "${RESULT_DIR}" "$(dirname "${CACHE_DIR}")"

# PVTv2 weights (download with download_and_inspect_pvt_weights.sh)
export POLYP_PVT_WEIGHTS=${POLYP_PVT_WEIGHTS:-/home/user1/aniket/Patchcore/PatchCore/models/PolypPVT.pth}
export PVTV2_B2_WEIGHTS=${PVTV2_B2_WEIGHTS:-/home/user1/aniket/Patchcore/PatchCore/models/pvt_v2_b2.pth}

# On macOS, faiss-cpu + torch clash on OpenMP; harmless on the GPU cluster.
export KMP_DUPLICATE_LIB_OK=${KMP_DUPLICATE_LIB_OK:-TRUE}

export PYTHONPATH=src
FORCE=${FORCE:-0}

echo "=================================================================="
echo " Streaming PatchCore | backbone=${BACKBONE} drift=${DRIFT}/${DRIFT_MODE}"
echo " data=${DATA_PATH}/${CLASSNAME}  M=${CAPACITY}  k=${N_NN}"
echo " cache=${CACHE_DIR}  results=${RESULT_DIR}"
echo "=================================================================="

# ─── STEP 1: cache embeddings (GPU) ───────────────────────────────────
echo -e "\n[1/5] Caching embeddings (frozen ${BACKBONE}) ..."
$PY bin/cache_embeddings.py \
  --backbone_name "${BACKBONE}" \
  "${LAYERS[@]}" \
  --data_path "${DATA_PATH}" \
  --classname "${CLASSNAME}" \
  --drift "${DRIFT}" \
  --drift_mode "${DRIFT_MODE}" \
  --seed "${SEED}" \
  --resize "${RESIZE}" \
  --imagesize "${IMAGESIZE}" \
  --pretrain_embed_dimension "${PRE_DIM}" \
  --target_embed_dimension "${TGT_DIM}" \
  --patchsize "${PATCHSIZE}" \
  --gpu 0 \
  --out_dir "${CACHE_DIR}" || { echo "caching failed"; exit 1; }

# ─── STEP 2: Gate 1 — headroom ────────────────────────────────────────
echo -e "\n[2/5] Gate 1 (headroom) ..."
$PY bin/run_gate1.py --cache_dir "${CACHE_DIR}" --capacity "${CAPACITY}" \
  --n_nn "${N_NN}" --out "${RESULT_DIR}/gate1.json"
GATE1=$?
if [ "${GATE1}" -ne 0 ] && [ "${FORCE}" != "1" ]; then
  echo "Gate 1 FAILED — static bank is not hurt by drift, so there is nothing to fix."
  echo "Set FORCE=1 to run the remaining steps anyway."
  exit 2
fi

# ─── STEP 3: Gate 2 — proxy validation ────────────────────────────────
echo -e "\n[3/5] Gate 2 (proxy validation) ..."
$PY bin/run_gate2.py --cache_dir "${CACHE_DIR}" --capacity "${CAPACITY}" \
  --n_nn "${N_NN}" --out "${RESULT_DIR}/gate2.json"
GATE2=$?
if [ "${GATE2}" -ne 0 ] && [ "${FORCE}" != "1" ]; then
  echo "Gate 2 FAILED — the label-free proxy does not track labeled AUROC."
  echo "Set FORCE=1 to run the remaining steps anyway."
  exit 2
fi

# ─── STEP 4: train PPO ────────────────────────────────────────────────
echo -e "\n[4/5] Training PPO maintenance policy ..."
$PY bin/train_ppo.py \
  --cache_dir "${CACHE_DIR}" \
  --capacity "${CAPACITY}" \
  --warmup "${WARMUP}" \
  --total_env_steps "${PPO_STEPS}" \
  --seed "${TRAIN_SEEDS}" \
  --out "${PPO_OUT}" \
  --eval_baselines || { echo "PPO training failed"; exit 1; }

# ─── STEP 5: benchmark all policies ───────────────────────────────────
echo -e "\n[5/5] Benchmarking policies ..."
$PY bin/run_streaming_baseline.py \
  --cache_dir "${CACHE_DIR}" \
  --capacity "${CAPACITY}" \
  --warmup "${WARMUP}" \
  --n_nn "${N_NN}" \
  --policies "${POLICIES}" \
  --ppo_path "${PPO_OUT}" \
  --seeds "${EVAL_SEEDS}" \
  --out "${RESULT_DIR}" || { echo "benchmark failed"; exit 1; }

echo -e "\n=================================================================="
echo " DONE. Artifacts under ${RESULT_DIR}/"
echo "   gate1.json / gate2.json   — gate outcomes"
echo "   ${PPO_OUT}                — trained PPO policy"
echo "   ${RESULT_DIR}/results.csv — per-stage AUROC/PRO for every policy"
echo "=================================================================="
