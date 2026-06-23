#!/bin/bash
#SBATCH --job-name=patchcore-mambavision
#SBATCH --partition=LocalQ
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --gres=shard:32
#SBATCH --ntasks=1
#SBATCH --output=/home/user1/aniket/Patchcore/logs/patchcore_mambavision%j.out
#SBATCH --error=/home/user1/aniket/Patchcore/logs/patchcore_mambavision%j.err
#SBATCH --cpus-per-task=32

cd /home/user1/aniket/Patchcore/PatchCore/patchcore-inspection
module load compilers/anaconda3-2024.06
module load libs/cuda-12.8
source /apps/compilers/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate patchcore

# Ignore ~/.local site-packages (a transformers 4.57 there shadows the conda env and, with
# torch 2.0.1, silently disables the PyTorch backend). Pin a torch-2.0-compatible transformers.
export PYTHONNOUSERSITE=1

# Compute nodes are offline — load the MambaVision weights/remote-code from the HF cache. Run
# download_hf_backbones.sh on the login node first to populate it.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

pip install 'transformers==4.46.3'

# MambaVision needs the CUDA selective-scan kernels. Install PREBUILT wheels matching this
# env (cu118 / torch2.0 / cp39) instead of compiling from source — source builds need nvcc and
# would mismatch the loaded cuda-12.8 module against torch's CUDA 11.8. ABI is auto-selected
# from torch. NOTE: cp39 is hardcoded for Python 3.9 — bump if the env's Python changes.
pip install einops
ABI=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
pip install --no-deps "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.2.0.post2/causal_conv1d-1.2.0.post2+cu118torch2.0cxx11abi${ABI}-cp39-cp39-linux_x86_64.whl"
pip install --no-deps "https://github.com/state-spaces/mamba/releases/download/v1.2.0.post1/mamba_ssm-1.2.0.post1+cu118torch2.0cxx11abi${ABI}-cp39-cp39-linux_x86_64.whl"

# Sanity checks
nvidia-smi
which python
python --version
python -c "import torch; print('torch OK | CUDA:', torch.cuda.is_available())"
python -c "import transformers, torch; print('transformers', transformers.__version__, '| torch backend OK:', transformers.is_torch_available())"
python -c "import faiss; res = faiss.StandardGpuResources(); print('faiss-gpu OK:', faiss.__version__)"

# MambaVision needs CUDA-built selective-scan kernels + einops. Fail early with a hint if missing.
python -c "import einops, mamba_ssm, causal_conv1d; print('mamba kernels OK')" || {
  echo "ERROR: MambaVision needs 'mamba-ssm', 'causal-conv1d' and 'einops' (CUDA build)."
  echo "Install in the GPU env:  pip install mamba-ssm causal-conv1d einops"
  exit 1
}

mkdir -p results
echo "Starting Kvasir PatchCore training with MambaVision-T backbone..."

# Backbone: -b mambavision_t (ImageNet MambaVision-T-1K, weights auto-downloaded from HF Hub).
# Layers stages.1 (160ch /8) + stages.2 (320ch /16) = analogue of resnet layer2+layer3.
env PYTHONPATH=src python bin/run_patchcore.py \
  --gpu 0 \
  --seed 0 \
  --save_patchcore_model \
  --log_group IM224_MambaVisionT_s12_P01 \
  --log_project Kvasir_Results \
  results \
  patch_core \
  -b mambavision_t \
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
  -d kvasir \
  mvtec \
  /home/user1/aniket/Patchcore/dataset/kvasir_patchcore
