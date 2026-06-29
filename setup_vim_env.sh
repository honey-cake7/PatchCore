#!/bin/bash
# Create the dedicated conda env for Vision Mamba (Vim).
#
# WHY A SEPARATE ENV: Vim needs a FORKED mamba_ssm ("mamba-1p1p1", bidirectional) whose package
# name collides with the standard mamba_ssm used by the MambaVision backbone — they cannot coexist.
#
# RUN THIS ON A NODE WITH INTERNET + AN NVIDIA GPU + the CUDA 11.8 toolkit (to build the kernels).
set -e
cd "$(dirname "$0")"

# module is a shell function that may be undefined in a non-interactive SLURM/child
# shell; don't let a missing `module` abort the script (set -e) before we create the env.
# The direct conda.sh source below is the reliable path to `conda`.
module load compilers/anaconda3-2024.06 2>/dev/null || true
module load libs/cuda-11.8 2>/dev/null || true   # match torch cu118 for nvcc — NOT cuda-12.8
source /apps/compilers/anaconda3-2024.06/etc/profile.d/conda.sh

# Idempotent: building the CUDA kernels takes ~10-20 min, so skip if the env already
# exists. Lets train_vim.sh call this on every submission without rebuilding each time.
if conda env list | grep -qE '^vim[[:space:]]'; then
  echo "vim env already exists — skipping build."
else
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

  # Pre-download Vim weights into the HF cache (compute nodes load weights offline at train
  # time). Cache warm-up only — must not abort an otherwise-successful build.
  python -c "from huggingface_hub import snapshot_download; print('cached ->', snapshot_download('hustvl/Vim-base-midclstok'))" || echo "warn: HF prefetch failed"

  # Final check.
  python -c "import torch, mamba_ssm, causal_conv1d, einops; print('vim env OK | torch', torch.__version__)"
fi
echo "Done. Train with:  conda activate vim && sbatch train_vim.sh  (or bash train_vim.sh)"
