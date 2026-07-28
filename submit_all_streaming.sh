#!/bin/bash
# Submit the streaming RL pipeline for HyperKvasir (polyp-pvt) and every MVTec
# class (wideresnet50) as independent SLURM jobs — they run in parallel up to
# the cluster's GPU capacity; the rest queue.
#
# Each job gets its own TAG (classname_backbone_drift), so caches, reward fits,
# policies and results never collide. Logs land in ../logs/streaming_<name>_<jobid>.
#
#   ./submit_all_streaming.sh                # submit everything
#   ONLY=hyperkvasir ./submit_all_streaming.sh
#   ONLY=mvtec ./submit_all_streaming.sh

DATASET_ROOT=${DATASET_ROOT:-/home/user1/aniket/Patchcore/dataset}
ONLY=${ONLY:-all}

submit() {
    local name=$1 backbone=$2 data_path=$3 classname=$4
    shift 4
    sbatch --job-name="stream-${name}" \
        --output="../logs/streaming_${name}_%j.log" \
        --error="../logs/streaming_${name}_%j.err" \
        --export=ALL,BACKBONE="${backbone}",DATA_PATH="${data_path}",CLASSNAME="${classname}" \
        "$@" run_streaming.sh
}

if [ "${ONLY}" = "all" ] || [ "${ONLY}" = "hyperkvasir" ]; then
    echo "Submitting HyperKvasir (polyp-pvt) ..."
    # --mem matches train_hyperkvasir.sh: the hyperkvasir stream is large
    submit hyperkvasir polyp-pvt "${DATASET_ROOT}/hyperkvasir_patchcore" hyperkvasir --mem=12G
fi

if [ "${ONLY}" = "all" ] || [ "${ONLY}" = "mvtec" ]; then
    MVTEC_CLASSES=(bottle cable capsule carpet grid hazelnut leather metal_nut
                   pill screw tile toothbrush transistor wood zipper)
    for cls in "${MVTEC_CLASSES[@]}"; do
        echo "Submitting MVTec/${cls} (wideresnet50) ..."
        submit "mvtec-${cls}" wideresnet50 "${DATASET_ROOT}/mvtec" "${cls}"
    done
fi

echo "All jobs submitted. Watch with: squeue -u \$USER"
