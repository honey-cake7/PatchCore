#!/bin/bash
#SBATCH --job-name=streaming-rl
#SBATCH --partition=LocalQ
#SBATCH --account=default
#SBATCH --gres=shard:6
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=../logs/streaming_output_%j.log
#SBATCH --error=../logs/streaming_error_%j.log
# ──────────────────────────────────────────────────────────────────────
# run_streaming.sh — dataset/backbone-agnostic streaming RL pipeline.
# Generalization of run_streaming_polyp_pvt.sh: pass BACKBONE, DATA_PATH and
# CLASSNAME; layers/patchsize are selected per backbone. Steps:
#   1. cache patch embeddings over a drift-ordered normal stream + per-stage tests
#      (skipped automatically when the cache already exists; RECACHE=1 to redo)
#   2. Gate 1  — headroom  (does drift hurt a static bank?)
#   3. Gate 2  — proxy validation (does the label-free reward track AUROC?)
#   3.5 fit proxy-reward weights offline
#   4. train the PPO/GPPO maintenance policy
#   5. benchmark all policies (baselines + PPO), writing per-stage AUROC/PRO
#
# RL is only justified if BOTH gates pass. By default this STOPS if a gate or
# the reward fit fails; pass FORCE=1 to run every step regardless.
#   BACKBONE=wideresnet50 DATA_PATH=.../mvtec CLASSNAME=bottle sbatch run_streaming.sh
#   BACKBONE=polyp-pvt DATA_PATH=.../hyperkvasir_patchcore CLASSNAME=hyperkvasir sbatch run_streaming.sh
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

# ------------------------------------------------------------------------------
# CUDA guard — the shard gres plugin can export an invalid CUDA_VISIBLE_DEVICES
# (CUDA error 101 "invalid device ordinal"), which silently pushes the whole
# pipeline onto CPU. Probe with a real tensor alloc (torch.cuda.is_available()
# alone can pass while device init fails), repair CVD if a GPU is physically
# present, and fail loudly otherwise. ALLOW_CPU=1 to run CPU-only on purpose.
# ------------------------------------------------------------------------------
ALLOW_CPU=${ALLOW_CPU:-0}
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES-<unset>}"
echo "SLURM_JOB_GPUS       : ${SLURM_JOB_GPUS-<unset>}"
echo "SLURM_STEP_GPUS      : ${SLURM_STEP_GPUS-<unset>}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || echo "nvidia-smi not found"

cuda_probe() {
    python -c "import torch; torch.zeros(1).cuda(); print('CUDA probe OK:', torch.cuda.get_device_name(0))" 2>&1
}
if ! PROBE_OUT=$(cuda_probe); then
    echo "CUDA probe FAILED:"
    echo "${PROBE_OUT}"
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        echo "*** GPUs are visible to nvidia-smi but not to torch."
        echo "*** Resetting CUDA_VISIBLE_DEVICES (was: '${CUDA_VISIBLE_DEVICES-<unset>}') -> 0 and re-probing."
        export CUDA_VISIBLE_DEVICES=0
        if ! PROBE_OUT=$(cuda_probe); then
            echo "CUDA probe still failing after reset:"
            echo "${PROBE_OUT}"
            if [ "${ALLOW_CPU}" != "1" ]; then
                echo "FATAL: no usable GPU (check --gres request vs node gres.conf). Set ALLOW_CPU=1 to force a CPU run."
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
# Python
# ------------------------------------------------------------------------------
export PYTHONPATH=${PKG_DIR}/src:${PYTHONPATH}

# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------
export TF_ENABLE_ONEDNN_OPTS=0
# Per-stage eval scoring on GPU (torch cdist; matches faiss up to float noise).
# Unset or set to "faiss" for byte-identical stock-PatchCore scoring.
export STREAMING_EVAL_BACKEND=torch
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
ALLOW_CPU=${ALLOW_CPU} python - <<'EOF' || { echo "environment verification failed"; exit 1; }
import os, sys
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
if not torch.cuda.is_available() and os.environ.get("ALLOW_CPU") != "1":
    print("FATAL: torch sees no CUDA device (set ALLOW_CPU=1 to run CPU-only).")
    sys.exit(1)
EOF

# ------------------------------------------------------------------------------
# CONFIG (edit these / pass as env vars)
# ------------------------------------------------------------------------------
DATA_PATH=${DATA_PATH:-/home/user1/aniket/Patchcore/dataset/kvasir_patchcore}
CLASSNAME=${CLASSNAME:-kvasir}                 # subfolder under DATA_PATH (mvtec-style)
BACKBONE=${BACKBONE:-polyp-pvt}                # wideresnet50 | polyp-pvt | pvtv2_b2
DRIFT=${DRIFT:-staged_abrupt_4}                # staged_abrupt_4 | staged_gradual_4 | staged_cyclic_4
DRIFT_MODE=${DRIFT_MODE:-synthetic}            # synthetic | real (metadata-ordered)
SEED=${SEED:-0}

# Feature-extraction config per backbone (stock configs from train_mvtec.sh /
# train_polyp_pvt_hyperkvasir.sh):
#   wideresnet50       : layer2+layer3, patchsize 3  (IM224_WR50_L2-3_PS-3)
#   polyp-pvt/pvtv2_b2 : norm2+norm3 (stride/8+/16), patchsize 6
case "${BACKBONE}" in
    wideresnet50)
        LAYERS=(-le layer2 -le layer3)
        PATCHSIZE=${PATCHSIZE:-3}
        ;;
    polyp-pvt|pvtv2_b2)
        LAYERS=(-le norm2 -le norm3)
        PATCHSIZE=${PATCHSIZE:-6}
        ;;
    *)
        echo "FATAL: unknown BACKBONE='${BACKBONE}' (expected wideresnet50 | polyp-pvt | pvtv2_b2)"
        exit 1
        ;;
esac
PRE_DIM=${PRE_DIM:-1024}
TGT_DIM=${TGT_DIM:-1024}
RESIZE=${RESIZE:-256}
IMAGESIZE=${IMAGESIZE:-224}

# streaming / RL settings
CAPACITY=${CAPACITY:-2000}                     # memory budget M
WARMUP=${WARMUP:-100}                          # warmup images for stage-0 bank + reward scales
N_NN=${N_NN:-5}                                # k for k-NN scoring (matches train_polyp_pvt.sh)
PPO_STEPS=${PPO_STEPS:-100000}
# GPPO preserves gradients on clipped samples (bigger effective steps), so run
# hot early and anneal cold: fast initial progress, small updates near the end.
PPO_LR=${PPO_LR:-1e-3}                         # starting learning rate
PPO_LR_END=${PPO_LR_END:-1e-5}                 # linear anneal target
ADV_MODE=${ADV_MODE:-grpo}                     # gae | grpo (group-relative, critic-free)
CLIP_MODE=${CLIP_MODE:-gppo}                   # clip | gppo (gradient-preserving)
CLIP_HIGH=${CLIP_HIGH:-}                       # optional decoupled upper epsilon
REWARD_FORM=${REWARD_FORM:-level}              # level | delta (potential-based shaping)
TRAIN_SEEDS=${TRAIN_SEEDS:-0}                  # PPO training seed(s)
EVAL_SEEDS=${EVAL_SEEDS:-0,1,2}               # benchmark eval seeds (disjoint from train ideally)
POLICIES=${POLICIES:-static,fifo,reservoir,streaming_greedy_coreset,periodic_coreset,ppo}

# Class list: space-separated; classes run SEQUENTIALLY in this one job.
# Per-class artifacts land in results/streaming/<class>_<backbone>_<drift>/ and
# a combined CSV + status summary is written/printed at the end of the job.
CLASSNAMES=${CLASSNAMES:-${CLASSNAME:-kvasir}}

# PVTv2 weights (download with download_and_inspect_pvt_weights.sh)
export POLYP_PVT_WEIGHTS=${POLYP_PVT_WEIGHTS:-${PROJECT_ROOT}/models/PolypPVT.pth}
export PVTV2_B2_WEIGHTS=${PVTV2_B2_WEIGHTS:-${PROJECT_ROOT}/models/pvt_v2_b2.pth}

FORCE=${FORCE:-0}
RECACHE=${RECACHE:-0}

BB_TAG=$(echo "${BACKBONE}" | tr -d '-')
SUMMARY_CSV=results/streaming/summary_${BB_TAG}_${DRIFT}.csv

CLIP_HIGH_ARG=""
[ -n "${CLIP_HIGH}" ] && CLIP_HIGH_ARG="--clip_high ${CLIP_HIGH}"

# A cache is only trustworthy if every manifest.json is present: the writer
# preallocates the memmap first and writes manifest.json LAST (commit marker),
# so a job cut off mid-caching leaves dirs without manifests. Directory
# existence alone must not skip step 1 — an interrupted cache would wedge the
# class until a manual RECACHE=1.
cache_complete() {
    local dir=$1 n=0 d
    [ -f "${dir}/stream/manifest.json" ] || return 1
    for d in "${dir}"/test/stage_*/; do
        [ -d "${d}" ] || continue
        n=$((n + 1))
        [ -f "${d%/}/manifest.json" ] || return 1
    done
    [ "${n}" -ge 1 ]
}

# ------------------------------------------------------------------------------
# Per-class pipeline (steps 1, 3.5, 4, 5). A failure aborts THIS class only;
# the loop moves on so one bad class can't strand the rest of the sweep.
# ------------------------------------------------------------------------------
run_one_class() {
    local CLASSNAME=$1
    local TAG=${CLASSNAME}_${BB_TAG}_${DRIFT}
    local CACHE_DIR=cache/${TAG}
    local RESULT_DIR=results/streaming/${TAG}
    local PPO_OUT=${RESULT_DIR}/ppo_${TAG}.pt
    mkdir -p "${RESULT_DIR}" "$(dirname "${CACHE_DIR}")"

    echo "========================================================="
    echo " [${CLASSNAME}] backbone=${BACKBONE} drift=${DRIFT}/${DRIFT_MODE}"
    echo " [${CLASSNAME}] data=${DATA_PATH}/${CLASSNAME}  M=${CAPACITY}  k=${N_NN}"
    echo " [${CLASSNAME}] cache=${CACHE_DIR}  results=${RESULT_DIR}"
    echo "========================================================="

    # STEP 1: cache embeddings (GPU) — skipped when a COMPLETE cache exists
    if [ "${RECACHE}" != "1" ] && cache_complete "${CACHE_DIR}"; then
        echo -e "\n[${CLASSNAME} 1/5] Complete cache at ${CACHE_DIR} — skipping (RECACHE=1 to redo)."
    else
        if [ -d "${CACHE_DIR}/stream" ] && [ "${RECACHE}" != "1" ]; then
            echo -e "\n[${CLASSNAME} 1/5] INCOMPLETE cache at ${CACHE_DIR} (interrupted run?) — re-caching."
        fi
        # start from a clean dir: partial memmaps / stale stage dirs from an
        # older config must not survive into the fresh cache
        rm -rf "${CACHE_DIR}"
        echo -e "\n[${CLASSNAME} 1/5] Caching embeddings (frozen ${BACKBONE}) ..."
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
            --out_dir                   "${CACHE_DIR}" || { echo "[${CLASSNAME}] caching failed"; return 1; }
    fi

    # STEP 3.5: fit proxy-reward weights offline (ranking validation vs AUROC)
    echo -e "\n[${CLASSNAME} 3.5/5] Fitting proxy-reward weights ..."
    python -u bin/fit_reward_weights.py --cache_dir "${CACHE_DIR}" --capacity "${CAPACITY}" --warmup "${WARMUP}" --n_nn "${N_NN}" --out "${RESULT_DIR}/reward_weights.json" || { echo "[${CLASSNAME}] reward-weight fit failed (rho below threshold?)"; [ "${FORCE}" = "1" ] || return 1; }

    # Without the fitted weights train/benchmark fall back to the DEFAULT
    # reward config (q_coef=0 — the misaligned reward).
    local REWARD_JSON_ARG=""
    [ -f "${RESULT_DIR}/reward_weights.json" ] && REWARD_JSON_ARG="--reward_json ${RESULT_DIR}/reward_weights.json"

    # STEP 4: train PPO
    echo -e "\n[${CLASSNAME} 4/5] Training PPO maintenance policy ..."
    python -u bin/train_ppo.py \
        --cache_dir        "${CACHE_DIR}" \
        --capacity         "${CAPACITY}" \
        --warmup           "${WARMUP}" \
        --total_env_steps  "${PPO_STEPS}" \
        --seed             "${TRAIN_SEEDS}" \
        --out              "${PPO_OUT}" \
        --adv_mode         "${ADV_MODE}" \
        --clip_mode        "${CLIP_MODE}" \
        ${CLIP_HIGH_ARG} \
        --lr               "${PPO_LR}" \
        --lr_end           "${PPO_LR_END}" \
        --reward_form      "${REWARD_FORM}" \
        ${REWARD_JSON_ARG} \
        --eval_baselines || { echo "[${CLASSNAME}] PPO training failed"; return 1; }

    # STEP 5: benchmark all policies
    echo -e "\n[${CLASSNAME} 5/5] Benchmarking policies ..."
    python -u bin/run_streaming_baseline.py \
        --cache_dir  "${CACHE_DIR}" \
        --capacity   "${CAPACITY}" \
        --warmup     "${WARMUP}" \
        --n_nn       "${N_NN}" \
        --policies   "${POLICIES}" \
        --ppo_path   "${PPO_OUT}" \
        --seeds      "${EVAL_SEEDS}" \
        ${REWARD_JSON_ARG} \
        --out        "${RESULT_DIR}" || { echo "[${CLASSNAME}] benchmark failed"; return 1; }

    echo "[${CLASSNAME}] DONE -> ${RESULT_DIR}/results.csv"
}

# ------------------------------------------------------------------------------
# Loop over classes, then combine per-class results into one CSV + summary
# ------------------------------------------------------------------------------
declare -a OK_CLASSES=() FAILED_CLASSES=()
for cls in ${CLASSNAMES}; do
    if run_one_class "${cls}"; then
        OK_CLASSES+=("${cls}")
    else
        FAILED_CLASSES+=("${cls}")
        echo "[${cls}] FAILED — continuing with remaining classes."
    fi
done

# One combined table across all classes (adds a leading "class" column), plus
# cross-class AVERAGES per policy (over classes & seeds) printed to the log and
# saved next to it as summary_*_mean.csv.
CLASSNAMES="${CLASSNAMES}" BB_TAG="${BB_TAG}" DRIFT="${DRIFT}" \
SUMMARY_CSV="${SUMMARY_CSV}" python - <<'EOF'
import csv, os
from collections import defaultdict

classes = os.environ["CLASSNAMES"].split()
bb, drift = os.environ["BB_TAG"], os.environ["DRIFT"]
out = os.environ["SUMMARY_CSV"]

rows = []
for c in classes:
    path = f"results/streaming/{c}_{bb}_{drift}/results.csv"
    if not os.path.exists(path):
        continue
    with open(path) as f:
        for row in csv.DictReader(f):
            row["class"] = c
            rows.append(row)
if not rows:
    print("no per-class results.csv found; nothing to combine")
    raise SystemExit(0)

fields = ["class", "policy", "seed", "stage", "image_auroc", "pixel_auroc", "pro"]
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
print(f"combined results -> {out}")

metrics = ["image_auroc", "pixel_auroc", "pro"]
policies = list(dict.fromkeys(r["policy"] for r in rows))
stages = sorted({r["stage"] for r in rows})
sums = defaultdict(float)
counts = defaultdict(int)
for r in rows:
    for m in metrics:
        try:
            v = float(r[m])
        except (TypeError, ValueError):
            continue
        for s in (r["stage"], "mean"):
            sums[(r["policy"], s, m)] += v
            counts[(r["policy"], s, m)] += 1

mean_csv = out.replace(".csv", "_mean.csv")
with open(mean_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["policy", "stage"] + metrics)
    for p in policies:
        for s in stages + ["mean"]:
            w.writerow([p, s] + [
                round(sums[(p, s, m)] / counts[(p, s, m)], 4) if counts[(p, s, m)] else ""
                for m in metrics
            ])

titles = {"image_auroc": "image AUROC", "pixel_auroc": "pixel AUROC", "pro": "PRO"}
n_classes = len({r["class"] for r in rows})
n_seeds = len({r["seed"] for r in rows})
name_w = max(len(p) for p in policies + ["policy"]) + 2
col_w = 8
cols = [f"stage{s}" for s in stages] + ["mean"]
inner = name_w + col_w * len(cols)

def cell(p, s, m):
    n = counts[(p, s, m)]
    return f"{sums[(p, s, m)] / n:.3f}" if n else "–"

def mean_of(p, m):
    n = counts[(p, "mean", m)]
    return sums[(p, "mean", m)] / n if n else float("-inf")

n_cls = f"{n_classes} class" + ("es" if n_classes != 1 else "")
n_sd = f"{n_seeds} seed" + ("s" if n_seeds != 1 else "")
print(f"\nCross-class averages — {n_cls} × {n_sd}, best policy first")
for m in metrics:
    print("┌" + ("─ " + titles[m] + " ").ljust(inner + 2, "─") + "┐")
    print("│ " + "policy".ljust(name_w) + "".join(c.rjust(col_w) for c in cols) + " │")
    print("├" + "─" * (inner + 2) + "┤")
    for p in sorted(policies, key=lambda p: mean_of(p, m), reverse=True):
        row = "".join(cell(p, s, m).rjust(col_w) for s in stages + ["mean"])
        print("│ " + p.ljust(name_w) + row + " │")
    print("└" + "─" * (inner + 2) + "┘")
print(f"per-policy cross-class averages -> {mean_csv}")
EOF

echo "========================================================="
echo " RUN SUMMARY (${BACKBONE}, ${DRIFT})"
for cls in "${OK_CLASSES[@]}"; do
    echo "   ${cls} : OK      results/streaming/${cls}_${BB_TAG}_${DRIFT}/"
done
for cls in "${FAILED_CLASSES[@]}"; do
    echo "   ${cls} : FAILED  (search this log for '[${cls}]')"
done
echo " Combined table : ${SUMMARY_CSV}"
echo " Class averages : ${SUMMARY_CSV%.csv}_mean.csv (also printed above)"
echo " Per class      : results/streaming/<class>_${BB_TAG}_${DRIFT}/{results.csv, ppo_*.pt, reward_weights.json}"
echo "========================================================="
[ ${#FAILED_CLASSES[@]} -eq 0 ] || exit 1
