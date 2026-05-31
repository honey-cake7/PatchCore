#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# train_kvasir_capsule.sh
# Train PatchCore on Kvasir-Capsule data (reformatted to MVTec layout)
#
# PREREQUISITE: Run prepare_dataset_capsule.py first to create the
# MVTec-compatible folder structure.
#
# The "trick": we use `mvtec` as the dataset name because our folder
# structure is identical to MVTec-AD. PatchCore's MVTecDataset class
# reads it seamlessly. The subdataset name `capsule` corresponds to the
# capsule/ subfolder inside the data root.
# ──────────────────────────────────────────────────────────────────────

# ─── CONFIG (edit these) ─────────────────────────────────────────────
datapath=/home/user1/aniket/Patchcore/dataset/kvasir_capsule_patchcore
# ──────────────────────────────────────────────────────────────────────

datasets=('capsule')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

########################################################################
# Baseline: WideResNet-50, Layers 2 & 3, Coreset 10%, IM224
#
# This is a good starting configuration for medical imaging.
# Adjust --patchsize, coreset percentage (-p), and image size as needed.
########################################################################

python bin/run_patchcore.py \
    --gpu 0 \
    --seed 0 \
    --save_patchcore_model \
    --save_segmentation_images \
    --log_group KvasirCapsule_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 \
    --log_project KvasirCapsule_Results \
    results \
    patch_core \
        -b wideresnet50 \
        -le layer2 \
        -le layer3 \
        --faiss_on_gpu \
        --pretrain_embed_dimension 1024 \
        --target_embed_dimension 1024 \
        --anomaly_scorer_num_nn 1 \
        --patchsize 3 \
    sampler -p 0.1 approx_greedy_coreset \
    dataset \
        --resize 256 \
        --imagesize 224 \
        "${dataset_flags[@]}" \
        mvtec $datapath

echo ""
echo "Training complete! Check the 'results/' folder for outputs."
echo "Model saved under results/KvasirCapsule_Results/KvasirCapsule_WR50_*/"
