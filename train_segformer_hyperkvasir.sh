#!/bin/bash
#SBATCH --job-name=hyperkvasir-patchcore-segformer
#SBATCH --partition=LocalQ
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --gres=shard:32
#SBATCH --ntasks=1
#SBATCH --output=/home/user1/aniket/Patchcore/logs/hyperkvasir_segformer_%j.out
#SBATCH --error=/home/user1/aniket/Patchcore/logs/hyperkvasir_segformer_%j.err
#SBATCH --cpus-per-task=32

cd /home/user1/aniket/Patchcore/PatchCore/patchcore-inspection
module load compilers/anaconda3-2024.06
module load libs/cuda-12.8
source /apps/compilers/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate patchcore

# Ignore ~/.local site-packages (a transformers 4.57 there shadows the conda env and, with
# torch 2.0.1, silently disables the PyTorch backend). Install a torch-2.0-compatible pin.
export PYTHONNOUSERSITE=1

# Compute nodes are offline — load the SegFormer weights from the HF cache. Run
# download_hf_backbones.sh on the login node first to populate it.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

pip install 'transformers==4.46.3'

# Sanity checks
nvidia-smi
which python
python --version
python -c "import torch; print('torch OK | CUDA:', torch.cuda.is_available())"
python -c "import transformers, torch; print('transformers', transformers.__version__, '| torch backend OK:', transformers.is_torch_available())"
python -c "from transformers import SegformerModel; print('SegformerModel import OK')"
python -c "import faiss; res = faiss.StandardGpuResources(); print('faiss-gpu OK:', faiss.__version__)"

mkdir -p results
echo "Starting HyperKvasir PatchCore training with SegFormer MiT-b3 backbone..."

# Backbone: -b segformer_mit_b3 (ImageNet MiT-b3 encoder, weights auto-downloaded from HF Hub).
# Layers stages.1 (128ch /8) + stages.2 (320ch /16) = analogue of resnet layer2+layer3.
env PYTHONPATH=src python bin/run_patchcore.py \
  --gpu 0 \
  --seed 0 \
  --save_patchcore_model \
  --log_group IM224_SegFormerB3_s12_HyperKvasir \
  --log_project HyperKvasir_Results \
  results \
  patch_core \
  -b segformer_mit_b3 \
  -le stages.1 \
  -le stages.2 \
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
