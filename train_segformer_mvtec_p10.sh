#!/bin/bash
#SBATCH --job-name=patchcore-segformer-mvtec-p10
#SBATCH --partition=LocalQ
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --gres=shard:32
#SBATCH --ntasks=1
#SBATCH --output=/home/user1/aniket/Patchcore/logs/patchcore_segformer_mvtec_p10_%j.out
#SBATCH --error=/home/user1/aniket/Patchcore/logs/patchcore_segformer_mvtec_p10_%j.err
#SBATCH --cpus-per-task=32

# Stock PatchCore, SegFormer MiT-b3 (general ImageNet encoder, nvidia/mit-b3),
# at a TRUE 10% coreset. The earlier IM224_SegFormerB3_MVTec_S12_P01 run was
# mislabeled: its sampler flag was -p 0.2 (20%). This is the stock column for
# the budget-matched streaming comparison (CAPACITY_MODE=match CAPACITY_PCT=10).

cd /home/user1/aniket/Patchcore/PatchCore/patchcore-inspection
module load compilers/anaconda3-2024.06
module load libs/cuda-12.8
source /apps/compilers/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate patchcore

export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

pip install 'transformers==4.46.3'

nvidia-smi
which python
python --version
python -c "import torch; print('torch OK | CUDA:', torch.cuda.is_available())"
python -c "import transformers, torch; print('transformers', transformers.__version__, '| torch backend OK:', transformers.is_torch_available())"
python -c "from transformers import SegformerModel; print('SegformerModel import OK')"
python -c "import faiss; res = faiss.StandardGpuResources(); print('faiss-gpu OK:', faiss.__version__)"

mkdir -p results
echo "Starting MVTec PatchCore training with SegFormer MiT-b3 backbone (10% coreset)..."

datapath=/home/user1/aniket/Patchcore/dataset/mvtec
datasets=('bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut' 'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

env PYTHONPATH=src python bin/run_patchcore.py \
  --gpu 0 \
  --seed 0 \
  --save_patchcore_model \
  --log_group IM224_SegFormerB3_MVTec_S12_P01_10pct \
  --log_project MVTecAD_Results \
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
  sampler -p 0.1 approx_greedy_coreset \
  dataset \
  --resize 256 \
  --imagesize 224 \
  "${dataset_flags[@]}" \
  mvtec \
  "$datapath"
