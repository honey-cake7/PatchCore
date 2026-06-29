#!/bin/bash
# Create the dedicated conda env for Vision Mamba (Vim).
#
# WHY A SEPARATE ENV: Vim needs a FORKED mamba_ssm ("mamba-1p1p1", bidirectional) whose package
# name collides with the standard mamba_ssm used by the MambaVision backbone — they cannot coexist.
#
# RUN THIS ON A NODE WITH INTERNET + AN NVIDIA GPU + the CUDA 11.8 toolkit (to build the kernels).
set -e
cd "$(dirname "$0")"

module load compilers/anaconda3-2024.06
module load libs/cuda-11.8        # match torch cu118 for nvcc — NOT cuda-12.8
source /apps/compilers/anaconda3-2024.06/etc/profile.d/conda.sh

conda create -y -n vim python=3.10.13
conda activate vim
export PYTHONNOUSERSITE=1

# torch 2.1.1 + cu118 (Vim's pinned combo). faiss-gpu matched to the patchcore env.
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
conda install -y -c pytorch -c nvidia faiss-gpu=1.9.0

# PatchCore's own deps (click, scikit-image, einops, tqdm, ...). torch already satisfied.
pip install -r patchcore-inspection/requirements.txt
# Vim's vendored models_mamba.py expects timm 0.4.12 (old timm.models.registry/layers paths).
pip install 'timm==0.4.12'

# Vim's vendored CUDA kernels + forked mamba_ssm. Build against CUDA 11.8 (loaded above).
# Install causal_conv1d FIRST and pin it — installing mamba-1p1p1 can otherwise pull a newer
# causal_conv1d (1.2.x) and break Vim.
VIM_DIR="${VIM_DIR:-$HOME/Vim}"
[ -d "$VIM_DIR" ] || git clone https://github.com/hustvl/Vim.git "$VIM_DIR"
pip install -e "$VIM_DIR/causal_conv1d"
pip install --no-deps -e "$VIM_DIR/mamba-1p1p1"

# Pre-download Vim weights into the HF cache (compute nodes are offline).
python -c "from huggingface_hub import snapshot_download; print('cached ->', snapshot_download('hustvl/Vim-base-midclstok'))"

# Final check.
python -c "import torch, mamba_ssm, causal_conv1d, einops; print('vim env OK | torch', torch.__version__)"
echo "Done. Train with:  conda activate vim && sbatch train_vim.sh  (or bash train_vim.sh)"
