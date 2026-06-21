#!/bin/bash
#SBATCH --job-name=patchcore-gastronet
#SBATCH --partition=LocalQ
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --gres=shard:32
#SBATCH --ntasks=1
#SBATCH --output=/home/user1/aniket/Patchcore/logs/patchcore_gastro%j.out
#SBATCH --error=/home/user1/aniket/Patchcore/logs/patchcore_gastro%j.err
#SBATCH --cpus-per-task=32

cd /home/user1/aniket/Patchcore/PatchCore/patchcore-inspection
module load compilers/anaconda3-2024.06
module load libs/cuda-12.8
source /apps/compilers/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate patchcore

# Sanity checks
nvidia-smi
which python
python --version
python -c "import timm; print('timm OK')"
python -c "import torch; print('torch OK | CUDA:', torch.cuda.is_available())"
python -c "import faiss; res = faiss.StandardGpuResources(); print('faiss-gpu OK:', faiss.__version__)"

# Verify GastroNet weights file exists
python -c "
import torch
ckpt = torch.load('/home/user1/aniket/Patchcore/PatchCore/models/gastronet.pth', map_location='cpu')
print('Checkpoint type:', type(ckpt))
print('Checkpoint keys:', ckpt.keys() if isinstance(ckpt, dict) else 'raw state_dict')
"

mkdir -p results
echo "Starting Kvasir PatchCore training with GastroNet backbone..."

env PYTHONPATH=src python bin/run_patchcore.py \
  --gpu 0 \
  --seed 0 \
  --save_patchcore_model \
  --log_group IM224_GastroNet_L5-11_P01 \
  --log_project Kvasir_Results \
  results \
  patch_core \
  -b gastronet \
  -le blocks.5 \
  -le blocks.11 \
  --faiss_on_gpu \
  --pretrain_embed_dimension 768 \
  --target_embed_dimension 768 \
  --anomaly_scorer_num_nn 1 \
  --patchsize 3 \
  sampler -p 0.2 approx_greedy_coreset \
  dataset \
  --resize 256 \
  --imagesize 224 \
  -d kvasir \
  mvtec \
  /home/user1/aniket/Patchcore/dataset/kvasir_patchcore