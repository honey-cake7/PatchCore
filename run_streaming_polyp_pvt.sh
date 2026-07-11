#!/bin/bash
#SBATCH --job-name=streaming-polyppvt
#SBATCH --partition=LocalQ
#SBATCH --account=default
#SBATCH --gres=shard:40
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=../logs/streaming_output_%j.log
#SBATCH --error=../logs/streaming_error_%j.log
# ──────────────────────────────────────────────────────────────────────
# run_streaming_polyp_pvt.sh
# End-to-end RL memory-bank maintenance pipeline with the Polyp-PVT backbone:
#   1. cache patch embeddings over a drift-ordered normal stream + per-stage tests
#   2. Gate 1  — headroom  (does drift hurt a static bank?)
#   3. Gate 2  — proxy validation (does the label-free reward track AUROC?)
#   4. train the self-contained PPO maintenance policy
#   5. benchmark all policies (baselines + PPO), writing per-stage AUROC/PRO
#
# RL is only justified if BOTH gates pass. By default this STOPS if a gate fails;
# pass FORCE=1 to run every step regardless.
#   sbatch run_streaming_polyp_pvt.sh
#   FORCE=1 DRIFT=staged_gradual_4 sbatch run_streaming_polyp_pvt.sh
# ──────────────────────────────────────────────────────────────────────

# Go to submission directory
cd $SLURM_SUBMIT_DIR

# ------------------------------------------------------------------------------
# Project Paths
# ------------------------------------------------------------------------------
PROJECT_ROOT=${PROJECT_ROOT:-/home/user1/aniket/Patchcore/PatchCore}
PKG_DIR=${PKG_DIR:-${PROJECT_ROOT}/patchcore-inspection}
CONDA_ENV=${CONDA_ENV:-patchcore}

cd ${PKG_DIR}
mkdir -p logs

# ------------------------------------------------------------------------------
# Load Modules
# ------------------------------------------------------------------------------
module purge
module load compilers/anaconda3-2024.06
module load libs/cuda-12.8

# ------------------------------------------------------------------------------
# Activate Conda
# ------------------------------------------------------------------------------
source /apps/compilers/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# IMPORTANT: Put the environment Python first (non-interactive shells otherwise
# leave the base/system python ahead on PATH -> "python: command not found").
export PATH=$CONDA_PREFIX/bin:$(echo $PATH | sed "s#$CONDA_PREFIX/bin:##")
hash -r

# ------------------------------------------------------------------------------
# CUDA
# ------------------------------------------------------------------------------
export CUDA_HOME=/apps/libs/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
TORCH_LIB=$(python -c "import os,torch;print(os.path.join(os.path.dirname(torch.__file__),'lib'))" 2>/dev/null)
export LD_LIBRARY_PATH=${TORCH_LIB}:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ------------------------------------------------------------------------------
# Python
# ------------------------------------------------------------------------------
export PYTHONPATH=${PKG_DIR}/src:${PYTHONPATH}

# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------
export TF_ENABLE_ONEDNN_OPTS=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "========================================================="
echo "Python      : $(which python)"
echo "Version     : $(python --version)"
echo "Conda Env   : $CONDA_DEFAULT_ENV"
echo "CUDA_HOME   : $CUDA_HOME"
echo "========================================================="

# ------------------------------------------------------------------------------
# Verify Environment
# ------------------------------------------------------------------------------
python - <<'EOF'
import torch, faiss, timm, patchcore
print("="*60)
print("Environment OK")
print("Torch          :", torch.__version__)
print("CUDA           :", torch.version.cuda)
print("CUDA Available :", torch.cuda.is_available())
print("faiss          :", faiss.__version__)
print("timm           :", timm.__version__)
print("patchcore      : OK")
print("="*60)
EOF

# ------------------------------------------------------------------------------
# CONFIG (edit these)
# ------------------------------------------------------------------------------
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
export POLYP_PVT_WEIGHTS=${POLYP_PVT_WEIGHTS:-${PROJECT_ROOT}/models/PolypPVT.pth}
export PVTV2_B2_WEIGHTS=${PVTV2_B2_WEIGHTS:-${PROJECT_ROOT}/models/pvt_v2_b2.pth}

FORCE=${FORCE:-0}

echo "========================================================="
echo " Streaming PatchCore | backbone=${BACKBONE} drift=${DRIFT}/${DRIFT_MODE}"
echo " data=${DATA_PATH}/${CLASSNAME}  M=${CAPACITY}  k=${N_NN}"
echo " cache=${CACHE_DIR}  results=${RESULT_DIR}"
echo "========================================================="

# ------------------------------------------------------------------------------
# STEP 1: cache embeddings (GPU)
# ------------------------------------------------------------------------------
echo -e "\n[1/5] Caching embeddings (frozen ${BACKBONE}) ..."
python -u bin/cache_embeddings.py \
    --backbone_name             "${BACKBONE}" \
    "${LAYERS[@]}" \
    --data_path                 "${DATA_PATH}" \
    --classname                 "${CLASSNAME}" \
    --drift                     "${DRIFT}" \
    --drift_mode                "${DRIFT_MODE}" \
    --seed                      "${SEED}" \
    --resize                    "${RESIZE}" \
    --imagesize                 "${IMAGESIZE}" \
    --pretrain_embed_dimension  "${PRE_DIM}" \
    --target_embed_dimension    "${TGT_DIM}" \
    --patchsize                 "${PATCHSIZE}" \
    --gpu                       0 \
    --out_dir                   "${CACHE_DIR}" || { echo "caching failed"; exit 1; }

# ------------------------------------------------------------------------------
# STEP 2: Gate 1 — headroom
# ------------------------------------------------------------------------------
echo -e "\n[2/5] Gate 1 (headroom) ..."
python -u bin/run_gate1.py --cache_dir "${CACHE_DIR}" --capacity "${CAPACITY}" \
    --n_nn "${N_NN}" --out "${RESULT_DIR}/gate1.json"


# ------------------------------------------------------------------------------
# STEP 3: Gate 2 — proxy validation
# ------------------------------------------------------------------------------
echo -e "\n[3/5] Gate 2 (proxy validation) ..."
python -u bin/run_gate2.py --cache_dir "${CACHE_DIR}" --capacity "${CAPACITY}" \
    --n_nn "${N_NN}" --out "${RESULT_DIR}/gate2.json"


# ------------------------------------------------------------------------------
# STEP 4: train PPO
# ------------------------------------------------------------------------------
echo -e "\n[4/5] Training PPO maintenance policy ..."
python -u bin/train_ppo.py \
    --cache_dir        "${CACHE_DIR}" \
    --capacity         "${CAPACITY}" \
    --warmup           "${WARMUP}" \
    --total_env_steps  "${PPO_STEPS}" \
    --seed             "${TRAIN_SEEDS}" \
    --out              "${PPO_OUT}" \
    --eval_baselines || { echo "PPO training failed"; exit 1; }

# ------------------------------------------------------------------------------
# STEP 5: benchmark all policies
# ------------------------------------------------------------------------------
echo -e "\n[5/5] Benchmarking policies ..."
python -u bin/run_streaming_baseline.py \
    --cache_dir  "${CACHE_DIR}" \
    --capacity   "${CAPACITY}" \
    --warmup     "${WARMUP}" \
    --n_nn       "${N_NN}" \
    --policies   "${POLICIES}" \
    --ppo_path   "${PPO_OUT}" \
    --seeds      "${EVAL_SEEDS}" \
    --out        "${RESULT_DIR}" || { echo "benchmark failed"; exit 1; }

echo "========================================================="
echo " DONE. Artifacts under ${RESULT_DIR}/"
echo "   gate1.json / gate2.json   — gate outcomes"
echo "   ${PPO_OUT}                — trained PPO policy"
echo "   ${RESULT_DIR}/results.csv — per-stage AUROC/PRO for every policy"
echo "========================================================="
