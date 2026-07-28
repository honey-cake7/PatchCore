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
#
# CLASSNAMES is passed via the environment (--export=ALL): sbatch's
# --export=NAME=VALUE parsing splits on commas and would mangle a list value.

DATASET_ROOT=${DATASET_ROOT:-/home/user1/aniket/Patchcore/dataset}
ONLY=${ONLY:-all}

if [ "${ONLY}" = "all" ] || [ "${ONLY}" = "hyperkvasir" ]; then
    echo "Submitting HyperKvasir (polyp-pvt) ..."
    BACKBONE=polyp-pvt \
    DATA_PATH="${DATASET_ROOT}/hyperkvasir_patchcore" \
    CLASSNAMES="hyperkvasir" \
    sbatch --job-name=stream-hyperkvasir \
        --output="../logs/streaming_output_%j.log" \
        --error="../logs/streaming_error_%j.log" \
        --export=ALL --mem=12G run_streaming.sh
fi

if [ "${ONLY}" = "all" ] || [ "${ONLY}" = "mvtec" ]; then
    echo "Submitting MVTec (wideresnet50, 15 classes in one job) ..."
    BACKBONE=wideresnet50 \
    DATA_PATH="${DATASET_ROOT}/mvtec" \
    CLASSNAMES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper" \
    sbatch --job-name=stream-mvtec \
        --output="../logs/streaming_output_%j.log" \
        --error="../logs/streaming_error_%j.log" \
        --export=ALL run_streaming.sh
fi

echo "Submitted. Watch with: squeue -u \$USER"
