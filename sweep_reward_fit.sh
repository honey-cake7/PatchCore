#!/bin/bash
# Refit proxy-reward weights from SAVED traces across candidate forget_weight
# targets and report the fitted Spearman rho for each — no stream rollouts, so
# the whole sweep takes seconds. Run inside an activated patchcore env (e.g.
# an interactive shell after `conda activate patchcore`, or `sbatch --wrap`).
#
#   ./sweep_reward_fit.sh                                    # hyperkvasir default
#   TAG=bottle_wideresnet50_staged_abrupt_4 ./sweep_reward_fit.sh
#   WEIGHTS="0.25 0.5 1.0" ./sweep_reward_fit.sh
#
# Then retrain with the winner (step 1 skips, 3.5 refits from traces, 4-5 rerun):
#   FORGET_WEIGHT=<best> ONLY=hyperkvasir ./submit_all_streaming.sh

set -u
PROJECT_ROOT=${PROJECT_ROOT:-/home/user1/aniket/Patchcore/PatchCore}
PKG_DIR=${PKG_DIR:-${PROJECT_ROOT}/patchcore-inspection}
TAG=${TAG:-hyperkvasir_polyppvt_staged_abrupt_4}
WEIGHTS=${WEIGHTS:-0.25 0.5 0.75 1.0}

cd "${PKG_DIR}"
export PYTHONPATH=${PKG_DIR}/src:${PYTHONPATH:-}
RESULT_DIR=results/streaming/${TAG}
TRACES=${RESULT_DIR}/reward_traces.pkl
[ -f "${TRACES}" ] || { echo "no traces at ${TRACES} — run the pipeline (step 3.5) for this TAG first"; exit 1; }

echo "Refitting ${TAG} from ${TRACES}"
for w in ${WEIGHTS}; do
    echo "─── forget_weight=${w} ───────────────────────────────"
    python -u bin/fit_reward_weights.py \
        --traces_in     "${TRACES}" \
        --forget_weight "${w}" \
        --min_rho       0 \
        --out           "${RESULT_DIR}/reward_weights_fw${w}.json" \
        | grep -E "recommended weights|rho_ranking" \
        || { echo "refit failed for forget_weight=${w}"; exit 1; }
done
echo "──────────────────────────────────────────────────────"
echo "Candidates written to ${RESULT_DIR}/reward_weights_fw<w>.json"
echo "Pick the forget_weight with the highest fitted rho, then retrain:"
echo "  FORGET_WEIGHT=<best> ONLY=hyperkvasir ./submit_all_streaming.sh"
