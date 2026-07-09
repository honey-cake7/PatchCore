#!/bin/bash
#SBATCH --job-name=patchcore-polyp-pvt-mvtec
#SBATCH --partition=LocalQ
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --gres=shard:32
#SBATCH --ntasks=1
#SBATCH --output=/home/user1/aniket/Patchcore/logs/patchcore_polyppvt_mvtec%j.out
#SBATCH --error=/home/user1/aniket/Patchcore/logs/patchcore_polyppvt_mvtec%j.err
#SBATCH --cpus-per-task=32

cd /home/user1/aniket/Patchcore/PatchCore/patchcore-inspection
module load compilers/anaconda3-2024.06
module load libs/cuda-12.8
source /apps/compilers/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate patchcore

nvidia-smi
which python
python --version
python -c "import timm; print('timm OK')"
python -c "import torch; print('torch OK | CUDA:', torch.cuda.is_available())"
python -c "import faiss; res = faiss.StandardGpuResources(); print('faiss-gpu OK:', faiss.__version__)"

export POLYP_PVT_WEIGHTS=/home/user1/aniket/Patchcore/PatchCore/models/PolypPVT.pth
export PVTV2_B2_WEIGHTS=/home/user1/aniket/Patchcore/PatchCore/models/pvt_v2_b2.pth

python -c "
import os
import torch
assert os.path.exists('${POLYP_PVT_WEIGHTS}'), f'missing weights: ${POLYP_PVT_WEIGHTS}'
ckpt = torch.load('${POLYP_PVT_WEIGHTS}', map_location='cpu')
print('Checkpoint type:', type(ckpt))
print('Checkpoint keys:', list(ckpt.keys())[:5] if isinstance(ckpt, dict) else 'raw state_dict')
"

mkdir -p results
echo "Starting MVTec PatchCore training with Polyp-PVT (PVTv2-B2) backbone..."

datapath=/home/user1/aniket/Patchcore/dataset/mvtec
datasets=('bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut' 'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

env PYTHONPATH=src python bin/run_patchcore.py \
  --gpu 0 \
  --seed 0 \
  --save_patchcore_model \
  --log_group IM224_PolypPVT_MVTec_norm23_P01 \
  --log_project MVTecAD_Results \
  results \
  patch_core \
  -b polyp-pvt \
  -le norm2 \
  -le norm3 \
  --faiss_on_gpu \
  --pretrain_embed_dimension 1024 \
  --target_embed_dimension 1024 \
  --anomaly_scorer_num_nn 5 \
  --patchsize 6 \
  sampler -p 0.2 approx_greedy_coreset \
  dataset \
  --resize 256 \
  --imagesize 224 \
  "${dataset_flags[@]}" \
  mvtec \
  "$datapath"