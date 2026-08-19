#!/bin/bash
# Submit the streaming RL pipeline as TWO SLURM jobs:
#   1. HyperKvasir (polyp-pvt)  — one class
#   2. MVTec (wideresnet50)     — all 15 classes looped inside ONE job
#
# Each job ends with a RUN SUMMARY block in its log plus combined CSVs:
#   results/streaming/summary_<backbone>_<drift>.csv        (all rows, class column)
#   results/streaming/summary_<backbone>_<drift>_mean.csv   (cross-class averages)
# Per-class artifacts: results/streaming/<class>_<backbone>_<drift>/
#
#   ./submit_all_streaming.sh                # submit both
#   ONLY=hyperkvasir ./submit_all_streaming.sh
#   ONLY=mvtec ./submit_all_streaming.sh
#   ITERATE=1 ONLY=mvtec ./submit_all_streaming.sh   # iterated reward refit:
#       refit each class's reward with the previous round's trained-PPO traces
#       added to the fitting set, then retrain + re-benchmark. Needs one
#       normal pass first (checkpoints must exist).
#   CAPACITY_MODE=match CAPACITY_PCT=10 ONLY=mvtec ./submit_all_streaming.sh
#       per-class M = CAPACITY_PCT% of stream patches = the budget a stock
#       PatchCore coreset gets at that percentage, so the stage-0 column is
#       directly comparable to a stock baseline at the same p. Results/summary
#       get an _m<pct> suffix, so runs at different percentages coexist
#       (budget-vs-accuracy curves). Traces re-record per pct (slower run).
#       Sweep example (submit sequentially, ONE at a time):
#         for p in 1 2.5 5 10; do CAPACITY_MODE=match CAPACITY_PCT=$p \
#             ONLY=mvtec ./submit_all_streaming.sh; done  # wait between jobs!
#   MVTEC_BACKBONE=segformer_mit_b3 ONLY=mvtec ./submit_all_streaming.sh
#       run the mvtec job on a different backbone (default wideresnet50).
#       segformer needs `transformers` installed in the patchcore env and
#       nvidia/mit-b3 in the HF cache (./download_hf_backbones.sh, login node).
#       New backbone = new cache tag: step 1 re-caches all 15 classes first.
#
# CLASSNAMES is passed via the environment (--export=ALL): sbatch's
# --export=NAME=VALUE parsing splits on commas and would mangle a list value.

DATASET_ROOT=${DATASET_ROOT:-/home/user1/aniket/Patchcore/dataset}
ONLY=${ONLY:-all}
# Shared cluster: always shards, never a whole GPU. Adjust the count here
# (e.g. GRES=shard:12) rather than editing run_streaming.sh's SBATCH header.
GRES=${GRES:-shard:6}
# AFTER=<jobid> queues this submission behind an earlier job (afterany: it runs
# whether or not the predecessor succeeded, so one bad class can't strand a
# whole chain). Used by sweep_paper_runs.sh to serialize a multi-run sweep on a
# single GPU. Each submission prints `SUBMITTED_JOBID=<id>` for the caller.
AFTER=${AFTER:-}
DEP_ARG=()
[ -n "${AFTER}" ] && DEP_ARG=(--dependency="afterany:${AFTER}")

submitted() { echo "SUBMITTED_JOBID=$1"; }

if [ "${ONLY}" = "all" ] || [ "${ONLY}" = "hyperkvasir" ]; then
    echo "Submitting HyperKvasir (polyp-pvt) ..."
    submitted "$(BACKBONE=polyp-pvt \
    DATA_PATH="${DATASET_ROOT}/hyperkvasir_patchcore" \
    CLASSNAMES="hyperkvasir" \
    sbatch --parsable --job-name=stream-hyperkvasir \
        --output="../logs/streaming_output_%j.log" \
        --error="../logs/streaming_error_%j.log" \
        --gres="${GRES}" "${DEP_ARG[@]}" \
        --export=ALL --mem=12G run_streaming.sh)"
fi

if [ "${ONLY}" = "kvasir" ]; then
    echo "Submitting Kvasir (polyp-pvt) ..."
    # Kvasir is the long-stream dataset (2401 train images); run_streaming.sh's
    # defaults (WARMUP=100, 100k PPO steps) were tuned for it, so nothing is
    # overridden here. CAPACITY is whatever the caller sets (the paper uses a
    # fixed M rather than a matched percentage for this one).
    submitted "$(BACKBONE=polyp-pvt \
    DATA_PATH="${DATASET_ROOT}/kvasir_patchcore" \
    CLASSNAMES="kvasir" \
    sbatch --parsable --job-name=stream-kvasir \
        --output="../logs/streaming_output_%j.log" \
        --error="../logs/streaming_error_%j.log" \
        --gres="${GRES}" "${DEP_ARG[@]}" \
        --export=ALL --mem=12G run_streaming.sh)"
fi

if [ "${ONLY}" = "all" ] || [ "${ONLY}" = "mvtec" ]; then
    echo "Submitting MVTec (wideresnet50, 15 classes in one job) ..."
    # MVTec streams are tiny (60-391 train images vs kvasir's 2401), so the
    # kvasir-scale defaults misbehave: WARMUP=100 eats half the stream (and
    # exceeds toothbrush's 60 images entirely), and 100k PPO steps replays a
    # ~200-step episode hundreds of times. Scale them down; all overridable.
    # PPO_LR 1e-5 -> 1e-6 (2026-08-02 full run): training improves monotonically
    # on ~all classes and PPO tops its own proxy on ~12/15 — the sweep's 1e-4
    # was still too hot for tiny streams (iter-3 best-restores on half of them).
    submitted "$(BACKBONE="${MVTEC_BACKBONE:-wideresnet50}" \
    DATA_PATH="${DATASET_ROOT}/mvtec" \
    WARMUP="${WARMUP:-15}" \
    PPO_STEPS="${PPO_STEPS:-50000}" \
    PPO_LR="${PPO_LR:-1e-5}" \
    PPO_LR_END="${PPO_LR_END:-1e-6}" \
    CLASSNAMES="${CLASSNAMES:-bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper}" \
    sbatch --parsable --job-name=stream-mvtec \
        --output="../logs/streaming_output_%j.log" \
        --error="../logs/streaming_error_%j.log" \
        --gres="${GRES}" "${DEP_ARG[@]}" \
        --export=ALL run_streaming.sh)"
fi

echo "Submitted. Watch with: squeue -u \$USER"
