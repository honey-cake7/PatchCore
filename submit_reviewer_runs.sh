#!/bin/bash
# Every run the paper needs, from scratch, on the ι reward — CHAINED
# (--dependency=afterany) so the jobs execute one at a time on the shared GPU,
# then a CPU-only harvest that writes report.txt / report_short.txt / GAPS.md.
#
#   DRY_RUN=1 ./submit_reviewer_runs.sh            # print the plan, submit nothing
#   ./submit_reviewer_runs.sh                       # submit the whole chain
#   ONLY_RUN="kvasir_M2000 mvtec_m5" ./submit_reviewer_runs.sh
#   RUN_GROUPS="main pre" ./submit_reviewer_runs.sh # main | kappa | pre | backbone
#   AFTER=<jobid> ./submit_reviewer_runs.sh         # queue behind a running job
#   scancel -u $USER                                # cancel everything
#
# Reward: ι = mean log(S_bank / S_full-memory) over a whole-image held-out set
# (resampled every episode), dense per step, label-free, no fitted weights — so
# no calibration split, no permutation test, standard test splits everywhere.
# The teacher table is built once per cache (step 2 of run_streaming.sh).
#
# Groups:
#   main      5 budgets x 3 datasets, κ unlimited. Per run: 3 PPO training seeds
#             x 5 eval seeds; baselines static, fifo, reservoir, streaming greedy
#             coreset, periodic coreset, kcenter (Charikar), myopic_iota
#             (label-free one-step lookahead); offline coreset at exactly M
#             (phase-1 pool and whole stream); stage-0 test scored at 25/50/75/
#             100% of the stream plus every phase's own test set; ι logged at
#             each boundary. M=2000 runs add full-memory banks (the teacher —
#             the less-is-more check) and deployment latency / peak memory.
#   kappa     write cap κ ∈ {1,4,16,64} at M=2000 (κ=∞ is the main run) —
#             the accuracy-vs-writes Pareto curve.
#   pre       drift experiment: Kvasir + HyperKvasir at M=2000, PPO trained on
#             the pre-shift stream only, evaluated on every phase.
#   backbone  OFF by default (you asked to leave it out): ImageNet PVTv2-B2 on
#             Kvasir + HyperKvasir at M=2000 (re-embeds the stream, new cache tag).
# Toothbrush is excluded from MVTec: its whole stage 0 (15 images) is warmup.
#
# All results land in fresh *_iota* dirs; nothing existing is overwritten.
set -uo pipefail
cd "$(dirname "$0")"

DRY_RUN=${DRY_RUN:-0}
ONLY_RUN=${ONLY_RUN:-all}
RUN_GROUPS=${RUN_GROUPS:-main kappa pre}
HARVEST=${HARVEST:-1}
LAST_JOBID=${AFTER:-}

# ---- one protocol for every run -------------------------------------------
# PPO: smaller net, hotter LR, fewer epochs (user request). Untuned starting
# point — override any of these from the environment.
PROTOCOL=(
    REWARD_MODE=iota HELDOUT_FRAC=${HELDOUT_FRAC:-0.05}
    STAGE0_ONLY=0 STAGE0_EVERY=1 OFFLINE=1 ITERATE=0 CALIB_SPLIT=0
    TRAIN_SEEDS=${TRAIN_SEEDS:-0,1,2} EVAL_SEEDS=${EVAL_SEEDS:-0,1,2,3,4}
    HIDDEN=${HIDDEN:-64} EPOCHS=${EPOCHS:-2} ADV_MODE=grpo ADV_BASELINE=loo
)
MVTEC_LR=(PPO_LR=${MVTEC_PPO_LR:-3e-4} PPO_LR_END=${MVTEC_PPO_LR_END:-1e-5})
ENDO_LR=(PPO_LR=${ENDO_PPO_LR:-1e-3} PPO_LR_END=${ENDO_PPO_LR_END:-1e-5})
MVTEC_CLASSES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile transistor wood zipper"
ENDO_MEM=${ENDO_MEM:-48G}   # full-memory bank / teacher stream ≈ 8-11 GB host RAM

# name|group|ONLY|extra env (space separated, no spaces inside values)
CONFIGS=()
for b in "M1000|CAPACITY=1000" "M2000|CAPACITY=2000 FULL_MEMORY=1 DEPLOY=1" \
         "m2.5|CAPACITY_MODE=match CAPACITY_PCT=2.5" "m5|CAPACITY_MODE=match CAPACITY_PCT=5" \
         "m10|CAPACITY_MODE=match CAPACITY_PCT=10"; do
    IFS='|' read -r bt benv <<< "${b}"
    tag="_${bt}_iota"; [[ ${bt} == m* ]] && tag="_iota"   # match mode adds _m<pct> itself
    CONFIGS+=("mvtec_${bt}|main|mvtec|${benv} RESULT_TAG=${tag}")
done
for ds in kvasir hyperkvasir; do
    for b in "M1000|CAPACITY=1000" "M2000|CAPACITY=2000 FULL_MEMORY=1 DEPLOY=1" \
             "M4742|CAPACITY=4742" "M10000|CAPACITY=10000" "m1|CAPACITY_MODE=match CAPACITY_PCT=1"; do
        IFS='|' read -r bt benv <<< "${b}"
        tag="_${bt}_iota"; [[ ${bt} == m* ]] && tag="_iota"
        CONFIGS+=("${ds}_${bt}|main|${ds}|${benv} RESULT_TAG=${tag} MEM=${ENDO_MEM}")
    done
done
for ds in mvtec kvasir hyperkvasir; do
    mem=""; [ "${ds}" != "mvtec" ] && mem="MEM=${ENDO_MEM}"
    for k in 1 4 16 64; do
        CONFIGS+=("${ds}_M2000_k${k}|kappa|${ds}|CAPACITY=2000 KAPPA=${k} RESULT_TAG=_M2000_iota_k${k} ${mem}")
    done
done
for ds in kvasir hyperkvasir; do
    CONFIGS+=("${ds}_M2000_pre|pre|${ds}|CAPACITY=2000 PRESHIFT=1 RESULT_TAG=_M2000_iota_pre MEM=${ENDO_MEM}")
    CONFIGS+=("${ds}_M2000_pvtv2b2|backbone|${ds}|CAPACITY=2000 ENDO_BACKBONE=pvtv2_b2 FULL_MEMORY=1 RESULT_TAG=_M2000_iota MEM=${ENDO_MEM}")
done

wanted() {   # name group
    local r g ok=1
    for g in ${RUN_GROUPS}; do [ "${g}" = "$2" ] && ok=0; done
    [ ${ok} -eq 0 ] || return 1
    [ "${ONLY_RUN}" = "all" ] && return 0
    for r in ${ONLY_RUN}; do [ "${r}" = "$1" ] && return 0; done
    return 1
}

echo "========================================================="
echo " Paper runs (ι reward) — chained one at a time on the shared GPU"
echo " groups: ${RUN_GROUPS}"
echo " protocol: ${PROTOCOL[*]}"
[ "${DRY_RUN}" = "1" ] && echo " DRY_RUN=1 — nothing will be submitted"
echo "========================================================="

n=0
for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r name group only extra <<< "${cfg}"
    wanted "${name}" "${group}" || continue
    n=$((n + 1))
    ds_env=("${ENDO_LR[@]}")
    if [ "${only}" = "mvtec" ]; then
        ds_env=("${MVTEC_LR[@]}" CLASSNAMES="${MVTEC_CLASSES}")
    fi
    echo ""
    echo "--- ${name}  [${group}]  ONLY=${only}  ${extra}"
    if [ "${DRY_RUN}" = "1" ]; then
        echo "    would run: env ${PROTOCOL[*]} ${ds_env[*]} ${extra} AFTER=${LAST_JOBID:-<none>} ONLY=${only} ./submit_all_streaming.sh"
        LAST_JOBID="<dry>"
        continue
    fi
    # shellcheck disable=SC2086  # ${extra} is a deliberate word-split list
    OUT=$(env "${PROTOCOL[@]}" "${ds_env[@]}" ${extra} AFTER="${LAST_JOBID}" ONLY="${only}" \
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

# CPU-only harvest once the chain is done (no GPU request)
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
echo " Queued ${n} run(s). Watch: squeue -u \$USER   Cancel: scancel -u \$USER"
echo " Results: patchcore-inspection/results/streaming/<class>_<bb>_staged_abrupt_4{_M<M>_iota[_k<κ>|_pre],_m<pct>_iota}/"
echo "========================================================="
