#!/bin/bash
#SBATCH --job-name=streaming-rl
#SBATCH --partition=LocalQ
#SBATCH --account=default
# Shards, never a whole GPU — this cluster is shared. Shard allocation is
# occasionally flaky here (job 1173: CUDA error 101 under shard:40; job 1294:
# nvidia-smi found no devices at all), and both times the same request
# succeeded on a later submission. Override the count without editing this
# header via `GRES=shard:12 ./submit_all_streaming.sh` — sbatch's command-line
# --gres wins over #SBATCH.
#SBATCH --gres=shard:6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=../logs/streaming_output_%j.log
#SBATCH --error=../logs/streaming_error_%j.log

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
# What SLURM actually handed us. A request can be accepted and still allocate
# no gres (job 1294: TresPerNode=gres/shard:6 but AllocTRES had no gres), which
# looks identical to a driver problem unless you print this.
if command -v scontrol >/dev/null 2>&1 && [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "AllocTRES            : $(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null \
        | tr ' ' '\n' | grep -m1 '^AllocTRES=' || echo '<unknown>')"
fi
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || echo "nvidia-smi not found"

cuda_probe() {
    python -c "import torch; torch.zeros(1).cuda(); print('CUDA probe OK:', torch.cuda.get_device_name(0))" 2>&1
}
# `nvidia-smi -L` exits 0 even when it prints "No devices found.", so testing
# its exit status alone reports "GPUs visible to nvidia-smi" on a node that has
# none — and sends you hunting a torch/CVD bug that isn't there. Require a real
# "GPU <n>:" line.
gpu_present() {
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q '^GPU [0-9]'
}
if ! PROBE_OUT=$(cuda_probe); then
    echo "CUDA probe FAILED:"
    echo "${PROBE_OUT}"
    if gpu_present; then
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
            echo "FATAL: this allocation has NO GPU — nvidia-smi lists no devices."
            echo "       Not a torch/CUDA_VISIBLE_DEVICES problem: the job did not"
            echo "       get a device. Check 'squeue -u \$USER' for a job still"
            echo "       holding the GPU, and 'sinfo -o \"%N %G %t\"' for node gres"
            echo "       and state, then resubmit. ALLOW_CPU=1 forces a CPU run"
            echo "       (unusably slow for this pipeline)."
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
# train_polyp_pvt_hyperkvasir.sh / train_segformer_mvtec.sh):
#   wideresnet50       : layer2+layer3, patchsize 3  (IM224_WR50_L2-3_PS-3)
#   polyp-pvt/pvtv2_b2 : norm2+norm3 (stride/8+/16), patchsize 6
#   segformer_mit_b3   : stages.1+stages.2 (stride/8+/16), patchsize 6;
#                        needs `transformers` in the env + mit-b3 in the HF
#                        cache (download_hf_backbones.sh on a login node)
case "${BACKBONE}" in
    wideresnet50)
        LAYERS=(-le layer2 -le layer3)
        PATCHSIZE=${PATCHSIZE:-3}
        ;;
    polyp-pvt|pvtv2_b2)
        LAYERS=(-le norm2 -le norm3)
        PATCHSIZE=${PATCHSIZE:-6}
        ;;
    segformer_mit_b3)
        LAYERS=(-le stages.1 -le stages.2)
        PATCHSIZE=${PATCHSIZE:-6}
        ;;
    *)
        echo "FATAL: unknown BACKBONE='${BACKBONE}' (expected wideresnet50 | polyp-pvt | pvtv2_b2 | segformer_mit_b3)"
        exit 1
        ;;
esac
PRE_DIM=${PRE_DIM:-1024}
TGT_DIM=${TGT_DIM:-1024}
RESIZE=${RESIZE:-256}
IMAGESIZE=${IMAGESIZE:-224}

# streaming / RL settings
CAPACITY=${CAPACITY:-2000}                     # memory budget M
# fixed : use CAPACITY as-is for every class.
# match : per-class M = CAPACITY_PCT% of the class's total stream patches — the
#         budget a stock PatchCore CAPACITY_PCT% greedy coreset gets, making
#         the stage-0 column directly comparable to a stock baseline at that
#         percentage. Results land in per-percentage dirs/summaries (suffix
#         _m<pct>), so budget sweeps don't clobber each other.
CAPACITY_MODE=${CAPACITY_MODE:-fixed}
CAPACITY_PCT=${CAPACITY_PCT:-10}               # percent, decimals allowed (e.g. 2.5)
[ "${CAPACITY_MODE}" = "match10" ] && { CAPACITY_MODE=match; CAPACITY_PCT=10; }
RESULT_SUFFIX=""
[ "${CAPACITY_MODE}" = "match" ] && RESULT_SUFFIX="_m${CAPACITY_PCT}"
# Extra suffix for running a VARIANT at the same capacity (e.g. a different
# reward-fit target) without overwriting an existing run's weights,
# checkpoints, results.csv and summary. Reward traces live in RESULT_DIR too,
# so a fresh tag re-records them unless you copy the old pkl+cfg across first.
RESULT_TAG=${RESULT_TAG:-}
RESULT_SUFFIX="${RESULT_SUFFIX}${RESULT_TAG}"
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
# STAGE0_ONLY=1: the paper reports ONLY stage-0 image/pixel AUROC. Skips all
# work serving drifted stages 1-3 and the forgetting metric — caching (step
# 1: 1 test-set embed instead of 4), per-stage evaluation (steps 3.5/4/5: 1
# labeled scoring pass per episode instead of 5). Opt-in, default off: with
# it unset every existing result stays byte-identical. Does NOT touch episode
# length, rng draws, or PPO training — the policy still runs the full stream.
STAGE0_ONLY=${STAGE0_ONLY:-0}
STAGE0_ONLY_ARG=""
[ "${STAGE0_ONLY}" = "1" ] && STAGE0_ONLY_ARG="--stage0_only"

# Reward-fit ranking target:
#   DRIFTED_WEIGHT*mean(stage>=1 AUROC) + FORGET_WEIGHT*forgetting
#   + STAGE0_WEIGHT*stage-0 AUROC
# Defaults reproduce the historical target, which EXCLUDES stage 0 — set
# STAGE0_WEIGHT (and zero the others) to fit weights for the stage-0-vs-stock
# comparison instead. Re-targeting reuses saved traces: seconds, no re-record.
# Under STAGE0_ONLY=1 the recorded traces have no forget_auroc and no
# drifted-stage entries (fit_reward_weights raises if FORGET_WEIGHT/
# DRIFTED_WEIGHT are non-zero against them), so the defaults flip to a
# pure stage-0 target unless the caller explicitly overrides them.
if [ "${STAGE0_ONLY}" = "1" ]; then
    FORGET_WEIGHT=${FORGET_WEIGHT:-0.0}
    DRIFTED_WEIGHT=${DRIFTED_WEIGHT:-0.0}
    STAGE0_WEIGHT=${STAGE0_WEIGHT:-1.0}
else
    FORGET_WEIGHT=${FORGET_WEIGHT:-0.5}
    DRIFTED_WEIGHT=${DRIFTED_WEIGHT:-1.0}
    STAGE0_WEIGHT=${STAGE0_WEIGHT:-0.0}
fi
MIN_RHO=${MIN_RHO:-0.7}                        # reward fit fails below this Spearman rho
# Permutation null: refit the same grid against N shuffled targets and log the
# p-value, so every class's fit carries a significance stamp. The grid reaches
# rho ~0.8 on shuffled targets with ~12 traces, so MIN_RHO=0.7 alone cannot
# tell a real fit from noise. Costs seconds; 0 disables.
PERMUTE=${PERMUTE:-0}
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
# ITERATE=1 = iterated reward refit: fold the previous round's trained PPO
# traces into the reward fitting set before refitting + retraining (run a
# normal pass first so each class has a checkpoint).
ITERATE=${ITERATE:-0}

BB_TAG=$(echo "${BACKBONE}" | tr -d '-')
SUMMARY_CSV=results/streaming/summary_${BB_TAG}_${DRIFT}${RESULT_SUFFIX}.csv

CLIP_HIGH_ARG=""
[ -n "${CLIP_HIGH}" ] && CLIP_HIGH_ARG="--clip_high ${CLIP_HIGH}"

# A cache is only trustworthy if every manifest.json is present: the writer
# preallocates the memmap first and writes manifest.json LAST (commit marker),
# so a job cut off mid-caching leaves dirs without manifests. Directory
# existence alone must not skip step 1 — an interrupted cache would wedge the
# class until a manual RECACHE=1.
#
# A stage0_only cache only has a stage_0/ test dir, which already satisfies
# the "n -ge 1" check below — so a stage0_only cache correctly reads as
# complete on a repeat stage0_only run (no endless re-cache). The opposite
# direction is the dangerous one: a stage0_only cache silently reused by a
# full (STAGE0_ONLY=0) run would look "complete" too, only ever score stage
# 0, and leave stages 1-3 permanently missing. cache_embeddings.py records
# stage0_only in every manifest.json precisely so this is detectable — a
# full run whose cache is flagged stage0_only=true is treated as incomplete.
cache_complete() {
    local dir=$1 want_stage0_only=$2 n=0 d
    [ -f "${dir}/stream/manifest.json" ] || return 1
    if [ "${want_stage0_only}" != "1" ] \
        && grep -q '"stage0_only": true' "${dir}/stream/manifest.json" 2>/dev/null; then
        return 1
    fi
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
    # results are per-percentage in match mode (cache is capacity-independent)
    local RESULT_DIR=results/streaming/${TAG}${RESULT_SUFFIX}
    local PPO_OUT=${RESULT_DIR}/ppo_${TAG}.pt
    # shadow the global so a match10 override cannot leak into the next class
    local CAPACITY=${CAPACITY}
    mkdir -p "${RESULT_DIR}" "$(dirname "${CACHE_DIR}")"

    echo "========================================================="
    echo " [${CLASSNAME}] backbone=${BACKBONE} drift=${DRIFT}/${DRIFT_MODE}"
    echo " [${CLASSNAME}] data=${DATA_PATH}/${CLASSNAME}  M=${CAPACITY}  k=${N_NN}"
    echo " [${CLASSNAME}] cache=${CACHE_DIR}  results=${RESULT_DIR}"
    echo " [${CLASSNAME}] STAGE0_ONLY=${STAGE0_ONLY} (stage-0-only AUROC; drifted stages 1-3 + forgetting skipped)"
    echo "========================================================="

    # STEP 1: cache embeddings (GPU) — skipped when a COMPLETE cache exists
    if [ "${RECACHE}" != "1" ] && cache_complete "${CACHE_DIR}" "${STAGE0_ONLY}"; then
        echo -e "\n[${CLASSNAME} 1/5] Complete cache at ${CACHE_DIR} — skipping (RECACHE=1 to redo)."
    else
        if [ -d "${CACHE_DIR}/stream" ] && [ "${RECACHE}" != "1" ]; then
            echo -e "\n[${CLASSNAME} 1/5] INCOMPLETE cache at ${CACHE_DIR} (interrupted run?) — re-caching."
        fi
        # start from a clean dir: partial memmaps / stale stage dirs from an
        # older config must not survive into the fresh cache; reward traces
        # recorded against the old cache are stale with it
        rm -rf "${CACHE_DIR}"
        rm -f "${RESULT_DIR}/reward_traces.pkl" "${RESULT_DIR}/reward_traces.cfg"
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
            --out_dir                   "${CACHE_DIR}" \
            ${STAGE0_ONLY_ARG} || { echo "[${CLASSNAME}] caching failed"; return 1; }
    fi

    # Budget-matched capacity: a stock PatchCore p% greedy coreset keeps p% of
    # ALL training patches; give the streaming bank the same per-class budget
    # so the stage-0 comparison is apples-to-apples at that percentage.
    # Computed from the cache (total stream patches), so it must run after
    # step 1.
    if [ "${CAPACITY_MODE}" = "match" ]; then
        CAPACITY=$(python - "${CACHE_DIR}" "${CAPACITY_PCT}" <<'PYEOF'
import sys

import numpy as np

e = np.load(f"{sys.argv[1]}/stream/embeddings.npy", mmap_mode="r")
pct = float(sys.argv[2])
print(max(1, int(round(pct / 100.0 * e.shape[0] * e.shape[1]))))
PYEOF
        ) || { echo "[${CLASSNAME}] match capacity computation failed"; return 1; }
        echo "[${CLASSNAME}] CAPACITY_MODE=match -> M=${CAPACITY} (${CAPACITY_PCT}% of stream patches, stock-budget-matched)"
    fi

    # STEP 3.5: fit proxy-reward weights offline (ranking validation vs AUROC).
    # Saved traces make refits (e.g. a new FORGET_WEIGHT) take seconds instead
    # of replaying every baseline rollout — but traces bake in warmup/capacity/
    # stage0_only (stage0_only traces carry no forget_auroc/drifted-stage
    # entries), so only reuse them when those still match (sidecar check).
    local TRACES_ARG=""
    if [ -f "${RESULT_DIR}/reward_traces.pkl" ] \
        && [ "$(cat "${RESULT_DIR}/reward_traces.cfg" 2>/dev/null)" = "${WARMUP}:${CAPACITY}:${STAGE0_ONLY}" ]; then
        TRACES_ARG="--traces_in ${RESULT_DIR}/reward_traces.pkl"
    fi
    # ITERATE=1: iterated reward refit — replay the previously trained PPO
    # policy, add its traces to the fitting set (persisted into the pkl), and
    # fit on baselines+PPO. Closes proxy directions the heuristic fitting set
    # never visited (reward hacking, e.g. proxy-optimal policies whose
    # forgetting collapses). Requires a prior round's checkpoint.
    local PPO_ITER_ARG=""
    if [ "${ITERATE}" = "1" ] && [ -f "${PPO_OUT}" ]; then
        echo "[${CLASSNAME}] ITERATE=1 — adding trained-PPO traces to the reward fit"
        PPO_ITER_ARG="--ppo_pt ${PPO_OUT}"
    elif [ "${ITERATE}" = "1" ]; then
        echo "[${CLASSNAME}] ITERATE=1 but no checkpoint at ${PPO_OUT} — plain fit"
    fi
    echo -e "\n[${CLASSNAME} 3.5/5] Fitting proxy-reward weights (target: ${DRIFTED_WEIGHT}*drifted + ${FORGET_WEIGHT}*forgetting + ${STAGE0_WEIGHT}*stage0) ..."
    python -u bin/fit_reward_weights.py --cache_dir "${CACHE_DIR}" --capacity "${CAPACITY}" --warmup "${WARMUP}" --n_nn "${N_NN}" --forget_weight "${FORGET_WEIGHT}" --drifted_weight "${DRIFTED_WEIGHT}" --stage0_weight "${STAGE0_WEIGHT}" --min_rho "${MIN_RHO}" --permute "${PERMUTE}" ${TRACES_ARG} ${PPO_ITER_ARG} ${STAGE0_ONLY_ARG} --out "${RESULT_DIR}/reward_weights.json"
    echo "${WARMUP}:${CAPACITY}:${STAGE0_ONLY}" > "${RESULT_DIR}/reward_traces.cfg"
   

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
        ${STAGE0_ONLY_ARG} \
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
RESULT_SUFFIX="${RESULT_SUFFIX}" \
SUMMARY_CSV="${SUMMARY_CSV}" python - <<'EOF'
import csv, os
from collections import defaultdict

classes = os.environ["CLASSNAMES"].split()
bb, drift = os.environ["BB_TAG"], os.environ["DRIFT"]
suffix = os.environ.get("RESULT_SUFFIX", "")
out = os.environ["SUMMARY_CSV"]

rows = []
for c in classes:
    path = f"results/streaming/{c}_{bb}_{drift}{suffix}/results.csv"
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
# "final_forgetting" is a pseudo-stage (stage-0 test re-scored with the final
# bank) — keep it out of the stage columns and the stage mean, mirroring
# run_streaming_baseline's own summary.
FORGET = "final_forgetting"
stages = sorted({r["stage"] for r in rows if r["stage"] != FORGET})
sums = defaultdict(float)
counts = defaultdict(int)
for r in rows:
    for m in metrics:
        try:
            v = float(r[m])
        except (TypeError, ValueError):
            continue
        if v != v:  # NaN (e.g. pixel/pro can be absent for forgetting rows)
            continue
        keys = (FORGET,) if r["stage"] == FORGET else (r["stage"], "mean")
        for s in keys:
            sums[(r["policy"], s, m)] += v
            counts[(r["policy"], s, m)] += 1

mean_csv = out.replace(".csv", "_mean.csv")
with open(mean_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["policy", "stage"] + metrics)
    for p in policies:
        for s in stages + [FORGET, "mean"]:
            w.writerow([p, "forget" if s == FORGET else s] + [
                round(sums[(p, s, m)] / counts[(p, s, m)], 4) if counts[(p, s, m)] else ""
                for m in metrics
            ])

titles = {"image_auroc": "image AUROC", "pixel_auroc": "pixel AUROC", "pro": "PRO"}
n_classes = len({r["class"] for r in rows})
n_seeds = len({r["seed"] for r in rows})
col_keys = stages + [FORGET, "mean"]
cols = [f"stage{s}" for s in stages] + ["forget", "mean"]
name_w = max(len(p) for p in policies + ["policy"]) + 2
col_w = max(8, max(len(c) for c in cols) + 2)
inner = name_w + col_w * len(cols)

def cell(p, s, m):
    n = counts[(p, s, m)]
    return f"{sums[(p, s, m)] / n:.3f}" if n else "–"

def mean_of(p, m):
    n = counts[(p, "mean", m)]
    return sums[(p, "mean", m)] / n if n else float("-inf")

n_cls = f"{n_classes} class" + ("es" if n_classes != 1 else "")
n_sd = f"{n_seeds} seed" + ("s" if n_seeds != 1 else "")
print(f"\nCross-class averages — {n_cls} × {n_sd}, best stage-mean first")
for m in metrics:
    print("┌" + ("─ " + titles[m] + " ").ljust(inner + 2, "─") + "┐")
    print("│ " + "policy".ljust(name_w) + "".join(c.rjust(col_w) for c in cols) + " │")
    print("├" + "─" * (inner + 2) + "┤")
    for p in sorted(policies, key=lambda p: mean_of(p, m), reverse=True):
        row = "".join(cell(p, s, m).rjust(col_w) for s in col_keys)
        print("│ " + p.ljust(name_w) + row + " │")
    print("└" + "─" * (inner + 2) + "┘")
print(f"per-policy cross-class averages -> {mean_csv}")
EOF

echo "========================================================="
echo " RUN SUMMARY (${BACKBONE}, ${DRIFT})"
for cls in "${OK_CLASSES[@]}"; do
    echo "   ${cls} : OK      results/streaming/${cls}_${BB_TAG}_${DRIFT}${RESULT_SUFFIX}/"
done
for cls in "${FAILED_CLASSES[@]}"; do
    echo "   ${cls} : FAILED  (search this log for '[${cls}]')"
done
echo " Combined table : ${SUMMARY_CSV}"
echo " Class averages : ${SUMMARY_CSV%.csv}_mean.csv (also printed above)"
echo " Per class      : results/streaming/<class>_${BB_TAG}_${DRIFT}${RESULT_SUFFIX}/{results.csv, ppo_*.pt, reward_weights.json}"
echo "========================================================="
[ ${#FAILED_CLASSES[@]} -eq 0 ] || exit 1
