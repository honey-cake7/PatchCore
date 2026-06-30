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
if conda env list | grep -qE '^vim[[:space:]]' \
   && conda run -n vim python -c "import mamba_ssm, causal_conv1d, faiss; faiss.StandardGpuResources()" 2>/dev/null; then
  echo "vim env already exists and is complete — skipping build."
else
  conda env remove -y -n vim 2>/dev/null || true   # clear any half-built env so create succeeds
  conda create -y -n vim python=3.10.13
  conda activate vim
  export PYTHONNOUSERSITE=1

  # torch 2.1.1 + cu118 (Vim's pinned combo).
  pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118

  # PatchCore's own deps (click, scikit-image, einops, tqdm, ...). torch already satisfied.
  pip install -r patchcore-inspection/requirements.txt
  # Vim's vendored models_mamba.py expects timm 0.4.12 (old timm.models.registry/layers paths).
  pip install 'timm==0.4.12'

  # requirements.txt pulls faiss-cpu, which overwrites the shared `faiss` package and hides
  # StandardGpuResources. Drop it and install the GPU build LAST so `import faiss` is faiss-gpu.
  pip uninstall -y faiss-cpu
  conda install -y -c pytorch -c nvidia faiss-gpu=1.9.0

  # The node's default nvcc is CUDA 13.0, but torch is cu118 — torch's cpp_extension aborts the
  # kernel build on that mismatch. Install a matching CUDA 11.8 toolkit into the env and point
  # the build at it via CUDA_HOME; robust even when `module load libs/cuda-11.8` is a no-op in
  # this non-interactive shell. $CONDA_PREFIX/bin/nvcc then shadows the system CUDA 13.0.
  conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
  export CUDA_HOME="$CONDA_PREFIX"

  # CUDA 11.8 nvcc rejects host gcc >11, but the node ships gcc 15. Install gcc/g++ 11 into
  # the env and force both nvcc (via -ccbin) and the C++ step (CC/CXX) to use them.
  conda install -y -c conda-forge gcc_linux-64=11 gxx_linux-64=11
  export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
  export NVCC_PREPEND_FLAGS="-ccbin $CXX"

  # Vim's vendored CUDA kernels + forked mamba_ssm. Build against CUDA 11.8 (loaded above).
  # Install causal-conv1d FIRST, installing mamba-1p1p1 can otherwise pull a newer
  # causal_conv1d (1.2.x) and break Vim. NOTE: the dirs are hyphenated (causal-conv1d),
  # even though the import name is causal_conv1d.
  # --no-build-isolation: both setup.py files `import torch` at build time, which pip's
  # default isolated PEP-517 build env lacks. Build against the env's torch + ninja instead,
  # using the env's CUDA 11.8 nvcc (CUDA_HOME set above) to compile the kernels.
  # setuptools<70: torch 2.1.1's cpp_extension does `from pkg_resources import packaging`,
  # removed in setuptools >=70 (conda create ships 82) — pin an older one for the build.
  pip install ninja "setuptools<70" wheel
  VIM_DIR="${VIM_DIR:-$HOME/Vim}"
  [ -d "$VIM_DIR" ] || git clone https://github.com/hustvl/Vim.git "$VIM_DIR"
  pip install --no-build-isolation -e "$VIM_DIR/causal-conv1d"
  pip install --no-build-isolation --no-deps -e "$VIM_DIR/mamba-1p1p1"

  # Pre-download Vim weights into the HF cache (compute nodes load weights offline at train
  # time). Cache warm-up only — must not abort an otherwise-successful build.
  python -c "from huggingface_hub import snapshot_download; print('cached ->', snapshot_download('hustvl/Vim-base-midclstok'))" || echo "warn: HF prefetch failed"

  # Final check.
  python -c "import torch, mamba_ssm, causal_conv1d, einops; print('vim env OK | torch', torch.__version__)"
  python -c "import faiss; faiss.StandardGpuResources(); print('faiss-gpu OK:', faiss.__version__)"
fi
echo "Done. Train with:  conda activate vim && sbatch train_vim.sh  (or bash train_vim.sh)"
