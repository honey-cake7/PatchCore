#!/bin/bash
#SBATCH --job-name=hyperkvasir-patchcore-polyppvt
#SBATCH --partition=LocalQ
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --gres=shard:32
#SBATCH --ntasks=1
#SBATCH --output=/home/user1/aniket/Patchcore/logs/hyperkvasir_polyppvt_%j.out
#SBATCH --error=/home/user1/aniket/Patchcore/logs/hyperkvasir_polyppvt_%j.err
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

# PVTv2-B2 weights. Download with download_and_inspect_pvt_weights.sh, then point
# these env vars at the .pth files (defaults are models/PolypPVT.pth, models/pvt_v2_b2.pth).
export POLYP_PVT_WEIGHTS=/home/user1/aniket/Patchcore/PatchCore/models/PolypPVT.pth
export PVTV2_B2_WEIGHTS=/home/user1/aniket/Patchcore/PatchCore/models/pvt_v2_b2.pth

# Verify the Polyp-PVT weights file exists and loads
python -c "
import torch
ckpt = torch.load('${POLYP_PVT_WEIGHTS}', map_location='cpu')
print('Checkpoint type:', type(ckpt))
print('Checkpoint keys:', list(ckpt.keys())[:5] if isinstance(ckpt, dict) else 'raw state_dict')
"

mkdir -p results
echo "Starting HyperKvasir PatchCore training with Polyp-PVT (PVTv2-B2) backbone..."

# Backbone: -b polyp-pvt (polyp fine-tuned) or -b pvtv2_b2 (ImageNet) to compare.
# Layers norm2 (stride/8, 128ch) + norm3 (stride/16, 320ch) = analogue of resnet layer2+layer3.
env PYTHONPATH=src python bin/run_patchcore.py \
  --gpu 0 \
  --seed 0 \
  --save_segmentation_images \
  --save_patchcore_model \
  --log_group IM224_PolypPVT_norm23_HyperKvasir \
  --log_project HyperKvasir_Results \
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
  -d hyperkvasir \
  mvtec \
  /home/user1/aniket/Patchcore/dataset/hyperkvasir_patchcore

echo "Training complete!"
