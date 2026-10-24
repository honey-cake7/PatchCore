#!/bin/bash
#SBATCH --job-name=reward-val
#SBATCH --partition=LocalQ
#SBATCH --account=default
#SBATCH --gres=shard:6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=../logs/reward_val_%j.log
#SBATCH --error=../logs/reward_val_%j.log
#
# Reward validation, no training: for each dataset record a fixed panel of
# maintenance rules (labelled per-stage AUROC + the raw label-free quantities
# candidate rewards are computed from), then print how each candidate reward
# ranks the banks against AUROC. Builds missing ι teacher tables first.
#
#   sbatch run_reward_validation.sh
#   DATASETS="kvasir" sbatch --export=ALL run_reward_validation.sh
#
# Results: patchcore-inspection/results/reward_validation/<tag>/  (re-analyze
# offline any time: python bin/validate_reward.py analyze --out <dir>)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT=${PROJECT_ROOT:-/home/user1/aniket/Patchcore/PatchCore}
DATA_ROOT=${DATA_ROOT:-/home/user1/aniket/Patchcore/dataset}
source /apps/compilers/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-patchcore}
export PATH=$CONDA_PREFIX/bin:$PATH
export PYTHONPATH=${PROJECT_ROOT}/patchcore-inspection/src:${PYTHONPATH:-}
export STREAMING_EVAL_BACKEND=torch
cd ${PROJECT_ROOT}/patchcore-inspection

# Pre-registered (critic round 1): seeds 0-3 endoscopy, 0-5 MVTec; held-out
# 5%, plus an MVTec 20% run (H-size sensitivity, reported only).
# name | cache tag | capacity | warmup | data subdir | classname | seeds | heldout
DATASETS=${DATASETS:-"kvasir screw capsule grid hyperkvasir screw_h20 capsule_h20 grid_h20"}
spec() {
    case "$1" in
        kvasir)      echo "kvasir_polyppvt_staged_abrupt_4 2000 100 kvasir_patchcore kvasir 0,1,2,3 0.05" ;;
        hyperkvasir) echo "hyperkvasir_polyppvt_staged_abrupt_4 2000 100 hyperkvasir_patchcore hyperkvasir 0,1,2,3 0.05" ;;
        *_h20)       echo "${1%_h20}_wideresnet50_staged_abrupt_4 2000 15 mvtec ${1%_h20} 0,1,2,3,4,5 0.20" ;;
        *)           echo "$1_wideresnet50_staged_abrupt_4 2000 15 mvtec $1 0,1,2,3,4,5 0.05" ;;
    esac
}

for ds in ${DATASETS}; do
    read -r TAG CAP WARM SUB CLS SEEDS HFRAC <<< "$(spec "$ds")"
    CACHE=cache/${TAG}
    OUT=results/reward_validation/${TAG}$([[ ${ds} == *_h20 ]] && echo _h20)
    echo "================ ${ds}: ${CACHE}  M=${CAP} warmup=${WARM} seeds=${SEEDS} H=${HFRAC} ================"
    if [ ! -f "${CACHE}/stream/manifest.json" ]; then
        echo "[${ds}] no cache at ${CACHE} — skipping"; continue
    fi
    if [ ! -f "${CACHE}/teacher_k5.npz" ]; then
        python -u bin/build_teacher.py --cache_dir "${CACHE}" --n_nn 5 \
            --data_path "${DATA_ROOT}/${SUB}" --classname "${CLS}" \
            || { echo "[${ds}] teacher build failed"; continue; }
    fi
    python -u bin/validate_reward.py record --cache_dir "${CACHE}" --capacity "${CAP}" \
        --warmup "${WARM}" --seeds "${SEEDS}" --heldout_frac "${HFRAC}" --out "${OUT}" \
        || { echo "[${ds}] record failed"; continue; }
    python -u bin/validate_reward.py analyze --out "${OUT}"
done

# Pre-registered verdict: selection on Kvasir + MVTec (5% H), confirmation on HyperKvasir
RV=results/reward_validation
MV=$(for c in screw capsule grid; do d=${RV}/${c}_wideresnet50_staged_abrupt_4; [ -f ${d}/summary.json ] && printf "%s," ${d}; done)
HK=${RV}/hyperkvasir_polyppvt_staged_abrupt_4
if [ -f ${RV}/kvasir_polyppvt_staged_abrupt_4/summary.json ]; then
    echo "================ VERDICT ================"
    python -u bin/validate_reward.py verdict --kvasir ${RV}/kvasir_polyppvt_staged_abrupt_4 \
        --mvtec "${MV%,}" $([ -f ${HK}/summary.json ] && echo --hyperkvasir ${HK})
fi
