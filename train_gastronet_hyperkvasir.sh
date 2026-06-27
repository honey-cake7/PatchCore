#!/bin/bash
#SBATCH --job-name=hyperkvasir-patchcore-gastronet
#SBATCH --partition=LocalQ
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --gres=shard:32
#SBATCH --ntasks=1
#SBATCH --output=/home/user1/aniket/Patchcore/logs/hyperkvasir_gastro_%j.out
#SBATCH --error=/home/user1/aniket/Patchcore/logs/hyperkvasir_gastro_%j.err
#SBATCH --cpus-per-task=32

cd /home/user1/aniket/Patchcore/patchcore-inspection
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
echo "Starting HyperKvasir PatchCore training with GastroNet backbone..."

env PYTHONPATH=src python bin/run_patchcore.py \
  --gpu 0 \
  --seed 0 \
  --save_patchcore_model \
  --log_group IM224_GastroNet_L5-11_HyperKvasir \
  --log_project HyperKvasir_Results \
  results \
  patch_core \
  -b gastronet \
  -le blocks.5 \
  -le blocks.8 \
  -le blocks.11 \
  --faiss_on_gpu \
  --pretrain_embed_dimension 1024 \
  --target_embed_dimension 1024 \
  --anomaly_scorer_num_nn 5 \
  --patchsize 6 \
  sampler -p 0.2 approx_greedy_coreset \
  dataset \
  --resize 256 \
  --imagesize 224 \
  -d hyperkvasir \
  mvtec \
  /home/user1/aniket/Patchcore/dataset/hyperkvasir_patchcore

echo "Training complete!"
