#!/bin/bash
#SBATCH --job-name=patchcore-vim
#SBATCH --partition=LocalQ
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --gres=shard:32
#SBATCH --ntasks=1
#SBATCH --output=/home/user1/aniket/Patchcore/logs/patchcore_vim%j.out
#SBATCH --error=/home/user1/aniket/Patchcore/logs/patchcore_vim%j.err
#SBATCH --cpus-per-task=32

# module loads are non-fatal (the `module` function may be undefined in this
# non-interactive shell); the direct conda.sh source is the reliable path to `conda`.
module load compilers/anaconda3-2024.06 2>/dev/null || true
module load libs/cuda-11.8 2>/dev/null || true   # Vim built against CUDA 11.8 (torch cu118)
source /apps/compilers/anaconda3-2024.06/etc/profile.d/conda.sh

# Build the dedicated Vim env once (forked mamba_ssm + CUDA kernels); reused on later runs.
if ! conda env list | grep -qE '^vim[[:space:]]'; then
  ( cd /home/user1/aniket/Patchcore/PatchCore/ && ./setup_vim_env.sh )
fi

cd /home/user1/aniket/Patchcore/PatchCore/patchcore-inspection
conda activate vim || {        # dedicated env, see setup_vim_env.sh
  echo "ERROR: 'vim' conda env missing, setup_vim_env.sh failed. Check the .out log."
  exit 1
}
export PYTHONNOUSERSITE=1

# Vim weights load from the HF cache (compute nodes are offline; setup_vim_env.sh pre-downloads).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Sanity checks
nvidia-smi
which python
python --version
python -c "import torch; print('torch OK | CUDA:', torch.cuda.is_available())"
python -c "import mamba_ssm, causal_conv1d, einops; print('vim deps OK')"
python -c "import faiss; res = faiss.StandardGpuResources(); print('faiss-gpu OK:', faiss.__version__)"

mkdir -p results
echo "Starting Kvasir PatchCore training with Vision Mamba (vim_base) backbone..."

# Vim is isotropic /16: stages.0 = layers.11, stages.1 = layers.17 (both 768ch, 14x14 maps).
# Single-scale, two depths — analogue of the gastronet ViT blocks.5/8/11 usage.
env PYTHONPATH=src python bin/run_patchcore.py \
  --gpu 0 \
  --seed 0 \
  --save_patchcore_model \
  --log_group IM224_VimBase_s01_P01 \
  --log_project Kvasir_Results \
  results \
  patch_core \
  -b vim_base \
  -le stages.0 \
  -le stages.1 \
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
