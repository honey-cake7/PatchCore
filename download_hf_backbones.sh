#!/bin/bash
# Pre-download the HuggingFace backbones (SegFormer mit-b3, MambaVision-T) into the HF cache.
#
# RUN THIS ON A NODE WITH INTERNET (e.g. the SLURM login node) BEFORE submitting
# train_segformer.sh / train_mambavision.sh. The compute nodes are offline and load the
# weights from the (shared-home) cache, because those jobs set HF_HUB_OFFLINE=1.
set -e
cd "$(dirname "$0")"

# Activate the same conda env the jobs use (adjust if you run this elsewhere).
module load compilers/anaconda3-2024.06 2>/dev/null || true
source /apps/compilers/anaconda3-2024.06/etc/profile.d/conda.sh 2>/dev/null || true
conda activate patchcore 2>/dev/null || true
export PYTHONNOUSERSITE=1

# snapshot_download fetches ALL repo files (config, weights, AND MambaVision's remote *.py)
# WITHOUT importing/building the model — so it runs fine on a GPU-less login node and never
# triggers MambaVision's CUDA-only mamba_ssm import.
python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("nvidia/mit-b3", "nvidia/MambaVision-T-1K"):
    path = snapshot_download(repo_id=repo)
    print(f"cached  {repo:28s} -> {path}")
PY

echo
echo "Done. Cache lives under \${HF_HOME:-~/.cache/huggingface} (shared home)."
echo "The train scripts set HF_HUB_OFFLINE=1, so the offline compute nodes load from it."
