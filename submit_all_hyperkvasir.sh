#!/bin/bash
# Submit all HyperKvasir PatchCore training jobs to SLURM

echo "Submitting WideResNet-50 (baseline) job..."
sbatch train_hyperkvasir.sh

echo "Submitting MambaVision job..."
sbatch train_mambavision_hyperkvasir.sh

echo "Submitting Polyp-PVT job..."
sbatch train_polyp_pvt_hyperkvasir.sh

echo "Submitting SegFormer job..."
sbatch train_segformer_hyperkvasir.sh

echo "Submitting GastroNet job..."
sbatch train_gastronet_hyperkvasir.sh

echo "All jobs submitted! Check queue status with: squeue"
