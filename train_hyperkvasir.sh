#!/bin/bash
#SBATCH --job-name=hyperkvasir-patchcore
#SBATCH --partition=LocalQ
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --gres=shard:32
#SBATCH --ntasks=1
#SBATCH --mem=120G
#SBATCH --output=/home/user1/aniket/Patchcore/logs/hyperkvasir_%j.out
#SBATCH --error=/home/user1/aniket/Patchcore/logs/hyperkvasir_%j.err
#SBATCH --cpus-per-task=32

cd /home/user1/aniket/Patchcore/patchcore-inspection

module load compilers/anaconda3-2024.06
module load libs/cuda-12.8
source /apps/compilers/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate patchcore

nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import faiss; res = faiss.StandardGpuResources(); print('faiss OK')"

mkdir -p results

echo "Starting HyperKvasir PatchCore training..."
env PYTHONPATH=src python bin/run_patchcore.py \
  --gpu 0 \
  --seed 0 \
  --save_patchcore_model \
  --log_group IM224_WR50_L2-3_HyperKvasir \
  --log_project HyperKvasir_Results \
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
  -d hyperkvasir \
  mvtec \
  /home/user1/aniket/Patchcore/dataset/hyperkvasir_patchcore

echo "Training complete!"