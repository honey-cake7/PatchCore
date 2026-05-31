#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# eval_kvasir_capsule.sh
# Evaluate a pretrained PatchCore model on Kvasir-Capsule test data.
#
# PREREQUISITE: Train a model first using train_kvasir_capsule.sh
# ──────────────────────────────────────────────────────────────────────

# ─── CONFIG (edit these) ─────────────────────────────────────────────
datapath=/home/user1/aniket/Patchcore/dataset/kvasir_capsule_patchcore
loadpath=results/KvasirCapsule_Results

# Pick the model folder from the training run
modelfolder=KvasirCapsule_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0
# ──────────────────────────────────────────────────────────────────────

savefolder=evaluated_results/${modelfolder}

datasets=('capsule')
model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/mvtec_'$dataset; done))
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

python bin/load_and_evaluate_patchcore.py \
    --gpu 0 \
    --seed 0 \
    --save_segmentation_images \
    $savefolder \
    patch_core_loader "${model_flags[@]}" --faiss_on_gpu \
    dataset \
        --resize 256 \
        --imagesize 224 \
        "${dataset_flags[@]}" \
        mvtec $datapath

echo ""
echo "Evaluation complete! Results saved to: ${savefolder}/"
echo ""
echo "NOTE: Since we use black masks (no pixel-level annotations),"
echo "only 'instance_auroc' (image-level) is meaningful."
echo "'full_pixel_auroc' and 'anomaly_pixel_auroc' will NOT be informative."
