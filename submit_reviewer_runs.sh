#!/bin/bash
# Submit every reviewer-checklist run that existing results cannot answer,
# CHAINED (--dependency=afterany) so they execute one at a time on the shared
# GPU, then a CPU-only harvest of everything once the chain finishes.
#
#   DRY_RUN=1 ./submit_reviewer_runs.sh          # print the plan, submit nothing
#   ./submit_reviewer_runs.sh                     # submit the whole chain
#   ONLY_RUN=kvasir_M2000 ./submit_reviewer_runs.sh
#   ONLY_RUN="mvtec_M1000 mvtec_5" ./submit_reviewer_runs.sh
#   AFTER=<jobid> ./submit_reviewer_runs.sh       # queue behind a running job
#
# One protocol for every run (checklist items 1-5, 7, 8), so budgets and
# datasets are comparable and nothing inherits the old runs' problems
# (ITERATE contamination at M=2000, mixed reward targets, test-set calibration):
#   CALIB_SPLIT=0.3   reward fitted on a stratified 30% calibration subset of
#                     the test set; every reported metric uses the other 70%
#   stage-0 reward target + permutation null (PERMUTE=200), ITERATE=0
#   EVAL_SEEDS=0..4   5 eval seeds
#   STAGE0_ONLY=0     all per-stage results (stages 1-3 + forgetting) kept
#   STAGE0_EVERY=1    + stage-0 test scored at 50% / 75% of the stream
#   OFFLINE=1         + offline greedy coreset at exactly M (phase-1 pool and
#                       whole-stream pool) as extra rows in results.csv
#   DEPLOY=1          + latency / peak GPU memory (M=2000 run of each dataset)
# Drift experiment (item 4): kvasir + hyperkvasir at M=2000 with PRESHIFT=1 —
#   reward fit, obs-norm and PPO training see the pre-shift stream only.
# NOT included (by request): the backbone check (ImageNet-only PVTv2/WRN-50 on
#   endoscopy) and the Polyp-PVT overlap count.
#
# Everything lands in FRESH dirs (..._M<M>_rv, ..._m<pct>_rv, ..._M2000_pre);
# no existing result is overwritten. Caches are reused (no re-embedding).
# Toothbrush is left out of MVTec: its stage 0 (15 images) is entirely warmup,
# so there is no stage-0 policy behaviour to measure.
set -uo pipefail
cd "$(dirname "$0")"

DRY_RUN=${DRY_RUN:-0}
ONLY_RUN=${ONLY_RUN:-all}
HARVEST=${HARVEST:-1}
LAST_JOBID=${AFTER:-}

PROTOCOL=(
    CALIB_SPLIT=0.3
    STAGE0_ONLY=0 STAGE0_WEIGHT=1 DRIFTED_WEIGHT=0 FORGET_WEIGHT=0
    PERMUTE=200 ITERATE=0
    EVAL_SEEDS=0,1,2,3,4
    STAGE0_EVERY=1 OFFLINE=1
)
MVTEC_CLASSES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile transistor wood zipper"
ENDO_MEM=${ENDO_MEM:-48G}   # whole-stream offline coreset pool is ~8-11 GB

# name|ONLY|extra env (space separated, no spaces inside values)
CONFIGS=(
    # ---- item 1: memory sweep, 5 budgets per dataset ----
    "mvtec_M1000|mvtec|CAPACITY=1000 RESULT_TAG=_M1000_rv"
    "mvtec_M2000|mvtec|CAPACITY=2000 RESULT_TAG=_M2000_rv DEPLOY=1"
    "mvtec_2.5|mvtec|CAPACITY_MODE=match CAPACITY_PCT=2.5 RESULT_TAG=_rv"
    "mvtec_5|mvtec|CAPACITY_MODE=match CAPACITY_PCT=5 RESULT_TAG=_rv"
    "mvtec_10|mvtec|CAPACITY_MODE=match CAPACITY_PCT=10 RESULT_TAG=_rv"
    "kvasir_M1000|kvasir|CAPACITY=1000 RESULT_TAG=_M1000_rv MEM=${ENDO_MEM}"
    "kvasir_M2000|kvasir|CAPACITY=2000 RESULT_TAG=_M2000_rv MEM=${ENDO_MEM} DEPLOY=1"
    "kvasir_M4742|kvasir|CAPACITY=4742 RESULT_TAG=_M4742_rv MEM=${ENDO_MEM}"
    "kvasir_M10000|kvasir|CAPACITY=10000 RESULT_TAG=_M10000_rv MEM=${ENDO_MEM}"
    "kvasir_1|kvasir|CAPACITY_MODE=match CAPACITY_PCT=1 RESULT_TAG=_rv MEM=${ENDO_MEM}"
    "hyperkvasir_M1000|hyperkvasir|CAPACITY=1000 RESULT_TAG=_M1000_rv MEM=${ENDO_MEM}"
    "hyperkvasir_M2000|hyperkvasir|CAPACITY=2000 RESULT_TAG=_M2000_rv MEM=${ENDO_MEM} DEPLOY=1"
    "hyperkvasir_M4742|hyperkvasir|CAPACITY=4742 RESULT_TAG=_M4742_rv MEM=${ENDO_MEM}"
    "hyperkvasir_M10000|hyperkvasir|CAPACITY=10000 RESULT_TAG=_M10000_rv MEM=${ENDO_MEM}"
    "hyperkvasir_1|hyperkvasir|CAPACITY_MODE=match CAPACITY_PCT=1 RESULT_TAG=_rv MEM=${ENDO_MEM}"
    # ---- item 4: drift experiment, pre-shift-only calibration + training ----
    "kvasir_pre|kvasir|CAPACITY=2000 RESULT_TAG=_M2000_pre PRESHIFT=1 MEM=${ENDO_MEM}"
    "hyperkvasir_pre|hyperkvasir|CAPACITY=2000 RESULT_TAG=_M2000_pre PRESHIFT=1 MEM=${ENDO_MEM}"
)

wanted() {
    [ "${ONLY_RUN}" = "all" ] && return 0
    local r
    for r in ${ONLY_RUN}; do [ "${r}" = "$1" ] && return 0; done
    return 1
}

echo "========================================================="
echo " Reviewer runs — chained one at a time on the shared GPU"
echo " protocol: ${PROTOCOL[*]}"
[ "${DRY_RUN}" = "1" ] && echo " DRY_RUN=1 — nothing will be submitted"
echo "========================================================="

n=0
for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r name only extra <<< "${cfg}"
    wanted "${name}" || continue
    n=$((n + 1))
    cls_env=()
    [ "${only}" = "mvtec" ] && cls_env=(CLASSNAMES="${MVTEC_CLASSES}")
    echo ""
    echo "--- ${name}  (ONLY=${only}, ${extra})"
    if [ "${DRY_RUN}" = "1" ]; then
        echo "    would run: env ${PROTOCOL[*]} ${extra} AFTER=${LAST_JOBID:-<none>} ONLY=${only} ./submit_all_streaming.sh"
        LAST_JOBID="<dry>"
        continue
    fi
    # shellcheck disable=SC2086  # ${extra} is a deliberate word-split list
    OUT=$(env "${PROTOCOL[@]}" "${cls_env[@]}" ${extra} AFTER="${LAST_JOBID}" ONLY="${only}" \
              ./submit_all_streaming.sh)
    echo "${OUT}"
    JOBID=$(sed -n 's/^SUBMITTED_JOBID=//p' <<< "${OUT}" | tail -1)
    if [ -z "${JOBID}" ]; then
        echo "    FAILED to capture a job id — stopping so later runs do not start"
        echo "    unqueued and collide on the GPU."
        exit 1
    fi
    echo "    job ${JOBID}${LAST_JOBID:+ (after ${LAST_JOBID})}"
    LAST_JOBID="${JOBID}"
done

# CPU-only harvest of every result dir once the chain is done (no GPU request)
if [ "${HARVEST}" = "1" ] && [ "${n}" -gt 0 ]; then
    PKG_DIR=${PKG_DIR:-$(pwd)/patchcore-inspection}
    CMD="source /apps/compilers/anaconda3-2024.06/etc/profile.d/conda.sh && conda activate ${CONDA_ENV:-patchcore} && cd ${PKG_DIR} && KMP_DUPLICATE_LIB_OK=TRUE python -u bin/harvest_existing_results.py"
    if [ "${DRY_RUN}" = "1" ]; then
        echo ""; echo "--- harvest (after ${LAST_JOBID}): ${CMD}"
    else
        HJOB=$(sbatch --parsable --job-name=harvest --partition=LocalQ --account=default \
            --cpus-per-task=8 --mem=32G --dependency="afterany:${LAST_JOBID}" \
            --output="../logs/harvest_%j.log" --wrap "${CMD}")
        echo ""; echo "--- harvest job ${HJOB} (after ${LAST_JOBID}) -> patchcore-inspection/results/harvest/"
    fi
fi

echo ""
echo "========================================================="
echo " Queued ${n} run(s). Watch: squeue -u \$USER"
echo " Results: patchcore-inspection/results/streaming/<class>_<bb>_staged_abrupt_4{_M<M>_rv,_m<pct>_rv,_M2000_pre}/"
echo "========================================================="
