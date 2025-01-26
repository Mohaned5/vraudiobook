#!/bin/bash -l
#
######################  SLURM OPTIONS  ######################
#SBATCH --partition=gpu               # or whichever GPU partition you use
#SBATCH --constraint=a100_80g         # ensure we get an A100 80GB card
#SBATCH --gres=gpu:1                  # request 1 GPU
#SBATCH --nodes=1                     # (1 node)
#SBATCH --cpus-per-gpu=4             # example: 4 CPU cores per GPU
#SBATCH --mem=32G                     # example memory request
#SBATCH --time=48:00:00               # example time limit (hh:mm:ss or d-hh:mm)
#SBATCH --job-name=Panime
#SBATCH --output=/scratch/users/%u/%j.out
#
###################  END OF SLURM OPTIONS  ##################

module load cuda
source ~/miniconda3/etc/profile.d/conda.sh
cd /scratch_tmp/prj/inf_vr_audio_book/vraudiobook

conda activate panfusion

WANDB_NAME=PanFusion_LRFinder \
python main.py fit \
    --data=PanimeDataModule \
    --model=PanFusion \
    --model.lr=3e-6 \
    --trainer.max_epochs=10 \
    --data.batch_size=1 \
    --data.num_workers=1

