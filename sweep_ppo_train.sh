#!/bin/bash
#SBATCH --job-name=ppo-sweep
#SBATCH --partition=LocalQ
#SBATCH --account=default
#SBATCH --gres=shard:6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=../logs/streaming_output_%j.log
#SBATCH --error=../logs/streaming_error_%j.log
# ──────────────────────────────────────────────────────────────────────
# sweep_ppo_train.sh — sweep PPO training hyperparameters on a few MVTec
# classes to fix the tiny-stream optimization failure (training reward
# declines from iter 1; the saved "best" checkpoint is a near-untrained
# policy). Uses the EXISTING caches + fitted rewards — no re-caching, no
# production artifacts touched.
#
#   sbatch sweep_ppo_train.sh                     # bottle screw zipper × A-D
#   CLASSES="bottle" sbatch --export=ALL sweep_ppo_train.sh
#   CONFIGS="C:3e-4:100000" sbatch --export=ALL sweep_ppo_train.sh
#
# Gate (same criterion that certified kvasir): PPO's proxy return must reach
# max(fifo, periodic_coreset). Pick the config passing on the most classes
# (tie-break: fewer steps), then bake it into submit_all_streaming.sh and
# rerun ONLY=mvtec.
# ──────────────────────────────────────────────────────────────────────

# Go to submission directory
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

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

# CUDA guard — same as run_streaming.sh (shard gres can export an invalid
# CUDA_VISIBLE_DEVICES, silently pushing everything onto CPU).
ALLOW_CPU=${ALLOW_CPU:-0}
cuda_probe() {
    python -c "import torch; torch.zeros(1).cuda(); print('CUDA probe OK:', torch.cuda.get_device_name(0))" 2>&1
}
if ! PROBE_OUT=$(cuda_probe); then
    echo "CUDA probe FAILED:"
    echo "${PROBE_OUT}"
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        echo "*** GPUs visible to nvidia-smi but not torch — resetting CUDA_VISIBLE_DEVICES -> 0."
        export CUDA_VISIBLE_DEVICES=0
        if ! PROBE_OUT=$(cuda_probe); then
            echo "${PROBE_OUT}"
            if [ "${ALLOW_CPU}" != "1" ]; then
                echo "FATAL: no usable GPU. Set ALLOW_CPU=1 to force a CPU run."
                exit 1
            fi
            echo "ALLOW_CPU=1 set — continuing on CPU."
        else
            echo "${PROBE_OUT}"
        fi
    else
        if [ "${ALLOW_CPU}" != "1" ]; then
            echo "FATAL: no GPU on this node/allocation. Set ALLOW_CPU=1 to force a CPU run."
            exit 1
        fi
        echo "ALLOW_CPU=1 set — continuing on CPU."
    fi
else
    echo "${PROBE_OUT}"
fi

# ------------------------------------------------------------------------------
# Python / Environment
# ------------------------------------------------------------------------------
export PYTHONPATH=${PKG_DIR}/src:${PYTHONPATH:-}
export TF_ENABLE_ONEDNN_OPTS=0
export STREAMING_EVAL_BACKEND=torch
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ------------------------------------------------------------------------------
# Sweep config (edit these / pass as env vars)
# ------------------------------------------------------------------------------
BB_TAG=${BB_TAG:-wideresnet50}
DRIFT=${DRIFT:-staged_abrupt_4}
CLASSES=${CLASSES:-bottle screw zipper}
CAPACITY=${CAPACITY:-2000}
WARMUP=${WARMUP:-30}
SEED=${SEED:-0}
LR_END=${LR_END:-1e-5}
# name:starting_lr:total_env_steps  (A = the current MVTec production config)
CONFIGS=${CONFIGS:-A:1e-3:30000 B:1e-3:100000 C:3e-4:100000 D:1e-4:100000}

SWEEP_DIR=results/streaming/ppo_sweep_${BB_TAG}_${DRIFT}
mkdir -p "${SWEEP_DIR}"

# tee would otherwise mask train_ppo.py's exit status in the || handler
set -o pipefail

for CLASSNAME in ${CLASSES}; do
    TAG=${CLASSNAME}_${BB_TAG}_${DRIFT}
    CACHE_DIR=cache/${TAG}
    REWARD_JSON=results/streaming/${TAG}/reward_weights.json
    [ -f "${CACHE_DIR}/stream/manifest.json" ] || { echo "[${CLASSNAME}] no cache at ${CACHE_DIR} — skipping"; continue; }
    [ -f "${REWARD_JSON}" ] || { echo "[${CLASSNAME}] no fitted reward at ${REWARD_JSON} — skipping"; continue; }

    for SPEC in ${CONFIGS}; do
        IFS=: read -r NAME LR STEPS <<< "${SPEC}"
        LOG=${SWEEP_DIR}/${CLASSNAME}_${NAME}.log
        echo "─── ${CLASSNAME} cfg ${NAME}: lr ${LR} -> ${LR_END}, steps ${STEPS} ───"
        python -u bin/train_ppo.py \
            --cache_dir       "${CACHE_DIR}" \
            --capacity        "${CAPACITY}" \
            --warmup          "${WARMUP}" \
            --total_env_steps "${STEPS}" \
            --seed            "${SEED}" \
            --out             "${SWEEP_DIR}/${CLASSNAME}_${NAME}.pt" \
            --adv_mode        grpo \
            --clip_mode       gppo \
            --lr              "${LR}" \
            --lr_end          "${LR_END}" \
            --reward_form     level \
            --reward_json     "${REWARD_JSON}" \
            --eval_baselines 2>&1 | tee "${LOG}" \
            || { echo "[${CLASSNAME}] cfg ${NAME} FAILED (see ${LOG})"; }
    done
done

# ------------------------------------------------------------------------------
# Summary: per class × config, PPO proxy return vs the heuristic gate
# ------------------------------------------------------------------------------
python - "${SWEEP_DIR}" <<'PYEOF'
import glob, os, re, sys

sweep_dir = sys.argv[1]
runs = {}  # (cls, cfg) -> {policy: return}
for log in sorted(glob.glob(os.path.join(sweep_dir, "*.log"))):
    cls, cfg = os.path.basename(log)[:-4].rsplit("_", 1)
    text = open(log).read()
    if "=== proxy return" not in text:
        continue
    table = text.split("=== proxy return", 1)[1]
    vals = dict(re.findall(r"^(\w+)\s+(-?\d+\.\d+)\s*$", table, re.M))
    runs[(cls, cfg)] = {k: float(v) for k, v in vals.items()}

if not runs:
    print("no completed runs with a proxy table found in", sweep_dir)
    sys.exit(0)

classes = sorted({c for c, _ in runs})
cfgs = sorted({k for _, k in runs})
print("\nPPO proxy return vs gate = max(fifo, periodic_coreset); PASS = ppo >= gate")
for cls in classes:
    print(f"\n  {cls}")
    print(f"    {'cfg':4s} {'ppo':>10s} {'fifo':>10s} {'periodic':>10s} {'gate':>10s}  verdict")
    for cfg in cfgs:
        r = runs.get((cls, cfg))
        if not r or "ppo" not in r:
            print(f"    {cfg:4s} {'—':>10s}  (missing/failed)")
            continue
        gate = max(r.get("fifo", float("-inf")),
                   r.get("periodic_coreset", float("-inf")))
        verdict = "PASS" if r["ppo"] >= gate else "fail"
        print(f"    {cfg:4s} {r['ppo']:>10.4f} {r.get('fifo', float('nan')):>10.4f} "
              f"{r.get('periodic_coreset', float('nan')):>10.4f} {gate:>10.4f}  {verdict}")

print("\nPer-config PASS count (pick the winner; tie-break: fewer steps):")
for cfg in cfgs:
    n = 0
    for cls in classes:
        r = runs.get((cls, cfg))
        if r and "ppo" in r and r["ppo"] >= max(r.get("fifo", float("-inf")),
                                                r.get("periodic_coreset", float("-inf"))):
            n += 1
    print(f"  {cfg}: {n}/{len(classes)}")
PYEOF
echo
echo "Logs + checkpoints in ${PKG_DIR}/${SWEEP_DIR}"
echo "Next: set the winning PPO_LR/PPO_STEPS in submit_all_streaming.sh and rerun ONLY=mvtec"
