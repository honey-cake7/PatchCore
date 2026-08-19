#!/bin/bash
# Submit every run the paper needs, CHAINED so they execute one at a time on
# the single shared GPU.
#
#   ./sweep_paper_runs.sh              # submit all five, queued back-to-back
#   DRY_RUN=1 ./sweep_paper_runs.sh    # print what would be submitted
#   ONLY_RUN=mvtec_2.5 ./sweep_paper_runs.sh   # just one config
#
# Runs (all with the stage-0 reward target — the paper reports stage 0 only):
#   1. hyperkvasir  M = 2000 fixed
#   2. hyperkvasir  M = 1% of stream patches
#   3. mvtec        M = 2.5% of stream patches   (15 classes in one job)
#   4. kvasir       M = 2000 fixed
#   5. kvasir       M = 400  fixed
#
# Chaining uses `--dependency=afterany`, so a failed run does not strand the
# rest of the queue. Everything lands in FRESH result dirs (see RESULT_TAG
# below) — no existing results are overwritten.
#
# On drift: the paper reports no drift. Stage 0 is the pre-drift quarter of the
# stream, so the reported column is already drift-free; DRIFT stays at its
# default purely as the stream-ordering mechanism that defines where stage 0
# ends. Nothing here reports stages 1-3 or forgetting.
set -uo pipefail

cd "$(dirname "$0")"

DRY_RUN=${DRY_RUN:-0}
ONLY_RUN=${ONLY_RUN:-all}
RESULTS_DIR=patchcore-inspection/results/streaming

# Stage-0 reward target + a significance stamp on every class's fit.
# STAGE0_ONLY=1 additionally prunes the work that only served drifted stages:
# 1 labeled scoring pass per episode instead of 5, and (for any cache built
# fresh) 1 test-set embed instead of 4. Existing FULL caches are still reused
# as-is — cache_complete() treats them as satisfying a stage-0-only request —
# so this costs no re-caching. Set STAGE0_ONLY=0 here if you ever want the
# drifted-stage columns back in results.csv.
STAGE0_ENV=(
    STAGE0_ONLY="${STAGE0_ONLY:-1}"
    STAGE0_WEIGHT=1
    DRIFTED_WEIGHT=0
    FORGET_WEIGHT=0
    PERMUTE=200
    ITERATE=0
)

# name|ONLY|extra env (space separated)
CONFIGS=(
    "hyperkvasir_M2000|hyperkvasir|CAPACITY=2000 RESULT_TAG=_M2000_s0"
    "hyperkvasir_1|hyperkvasir|CAPACITY_MODE=match CAPACITY_PCT=1 RESULT_TAG=_s0"
    "mvtec_2.5|mvtec|CAPACITY_MODE=match CAPACITY_PCT=2.5 RESULT_TAG=_s0"
    "kvasir_M2000|kvasir|CAPACITY=2000 RESULT_TAG=_M2000_s0"
    "kvasir_M400|kvasir|CAPACITY=400 RESULT_TAG=_M400_s0"
)

# ---------------------------------------------------------------------------
# Reward traces are keyed on "WARMUP:CAPACITY" (run_streaming.sh's sidecar), so
# a fixed-capacity run can reuse traces already recorded at the same warmup and
# capacity instead of replaying 12 baseline episodes. Copying a MISMATCHED
# sidecar is harmless — run_streaming.sh compares it and re-records — so this
# is an optimisation, never a correctness risk.
# ---------------------------------------------------------------------------
seed_traces() {
    local src="$1" dst="$2"
    [ -f "${src}/reward_traces.pkl" ] || return 0
    [ -f "${src}/reward_traces.cfg" ] || return 0
    mkdir -p "${dst}"
    if [ -f "${dst}/reward_traces.pkl" ]; then
        echo "    traces: ${dst} already has traces — leaving them alone"
        return 0
    fi
    cp "${src}/reward_traces.pkl" "${src}/reward_traces.cfg" "${dst}/"
    echo "    traces: seeded ${dst} from ${src} ($(cat "${src}/reward_traces.cfg"))"
}

echo "========================================================="
echo " Paper run sweep — ${#CONFIGS[@]} configs, chained one at a time"
echo " target: stage-0 only (${STAGE0_ENV[*]})"
[ "${DRY_RUN}" = "1" ] && echo " DRY_RUN=1 — nothing will be submitted"
echo "========================================================="

LAST_JOBID=""
for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r name only extra <<< "${cfg}"
    if [ "${ONLY_RUN}" != "all" ] && [ "${ONLY_RUN}" != "${name}" ]; then
        continue
    fi

    echo ""
    echo "--- ${name}  (ONLY=${only}, ${extra})"

    # Seed traces for fixed-capacity runs whose untagged sibling already has
    # them recorded at the same capacity. Match-mode runs compute CAPACITY
    # inside the job (after step 1), so their traces cannot be seeded here.
    case "${extra}" in
        *CAPACITY=*)
            case "${only}" in
                hyperkvasir) src="${RESULTS_DIR}/hyperkvasir_polyppvt_staged_abrupt_4" ;;
                kvasir)      src="${RESULTS_DIR}/kvasir_polyppvt_staged_abrupt_4" ;;
                *)           src="" ;;
            esac
            tag=$(sed -n 's/.*RESULT_TAG=\([^ ]*\).*/\1/p' <<< "${extra}")
            [ -n "${src}" ] && [ "${DRY_RUN}" != "1" ] && seed_traces "${src}" "${src}${tag}"
            ;;
    esac

    if [ "${DRY_RUN}" = "1" ]; then
        echo "    would run: env ${STAGE0_ENV[*]} ${extra} AFTER=${LAST_JOBID:-<none>}" \
             "ONLY=${only} ./submit_all_streaming.sh"
        LAST_JOBID="<dry>"
        continue
    fi

    OUT=$(env "${STAGE0_ENV[@]}" ${extra} AFTER="${LAST_JOBID}" ONLY="${only}" \
              ./submit_all_streaming.sh)
    echo "${OUT}"
    JOBID=$(sed -n 's/^SUBMITTED_JOBID=//p' <<< "${OUT}" | tail -1)
    if [ -z "${JOBID}" ]; then
        echo "    FAILED to capture a job id — stopping the chain so later runs"
        echo "    do not start unqueued and collide on the GPU."
        exit 1
    fi
    echo "    job ${JOBID}${LAST_JOBID:+ (after ${LAST_JOBID})}"
    LAST_JOBID="${JOBID}"
done

echo ""
echo "========================================================="
echo " Queued. Watch:   squeue -u \$USER"
echo " Results land in: ${RESULTS_DIR}/<class>_<backbone>_<drift><suffix>/"
echo "   hyperkvasir_polyppvt_staged_abrupt_4_M2000_s0"
echo "   hyperkvasir_polyppvt_staged_abrupt_4_m1_s0"
echo "   <class>_wideresnet50_staged_abrupt_4_m2.5_s0   (x15)"
echo "   kvasir_polyppvt_staged_abrupt_4_M2000_s0"
echo "   kvasir_polyppvt_staged_abrupt_4_M400_s0"
echo "========================================================="
