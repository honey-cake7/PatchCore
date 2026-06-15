#!/bin/bash
#SBATCH --job-name=patchcore-kvasir-ttt4as
#SBATCH --partition=LocalQ
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --gres=shard:32
#SBATCH --ntasks=1
#SBATCH --output=/home/user1/aniket/Patchcore/logs/patchcore_kvasir_ttt4as_%j.out
#SBATCH --error=/home/user1/aniket/Patchcore/logs/patchcore_kvasir_ttt4as_%j.err
#SBATCH --cpus-per-task=8

# Navigate to the correct working directory
cd /home/user1/aniket/Patchcore/PatchCore/patchcore-inspection

# Load modules and activate environment
module load compilers/anaconda3-2024.06
module load libs/cuda-12.8
source /apps/compilers/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate patchcore

# ─── CONFIG ───────────────────────────────────────────────────────────
datapath=/home/user1/aniket/Patchcore/dataset/kvasir_patchcore
# ──────────────────────────────────────────────────────────────────────

datasets=('kvasir')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))


echo "Starting Kvasir TRAINING with TTT4AS..."

########################################################################
# WideResNet-50, Layers 2 & 3, Coreset 10%, IM224
# train_val_split 0.8 reserves 20% nominal data for the TTT4AS THR baseline.
########################################################################

env PYTHONPATH=src python bin/run_patchcore.py \
    --gpu 0 \
    --seed 0 \
    --save_patchcore_model \
    --save_segmentation_images \
    --ttt4as \
    --ttt4as_features wrn50 \
    --percentile 99.0 \
    --thr_sigma 3.0 \
    --log_group Kvasir_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 \
    --log_project Kvasir_Results \
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
        --train_val_split 0.8 \
        "${dataset_flags[@]}" \
        mvtec $datapath

echo ""
echo "Training complete! Check the 'results/' folder for outputs."
echo "Model saved under results/Kvasir_Results/Kvasir_WR50_*/"
