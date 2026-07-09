#!/bin/bash
# Submit all MVTec PatchCore training jobs to SLURM

echo "Submitting WideResNet-50 (baseline) job..."
sbatch train_mvtec.sh

echo "Submitting MambaVision job..."
sbatch train_mambavision_mvtec.sh

echo "Submitting Polyp-PVT job..."
sbatch train_polyp_pvt_mvtec.sh

echo "Submitting SegFormer job..."
sbatch train_segformer_mvtec.sh

echo "Submitting GastroNet job..."
sbatch train_gastronet_mvtec.sh

echo "Submitting PVTv2-B2 job..."
sbatch train_pvtv2_b2_mvtec.sh

echo "All jobs submitted! Check queue status with: squeue"
