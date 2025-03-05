#!/bin/bash -l
#
######################  SLURM OPTIONS  ######################
#SBATCH --partition=gpu              
#SBATCH --gres=gpu:1               
#SBATCH --nodes=1             
#SBATCH --cpus-per-gpu=4           
#SBATCH --mem=32G                  
#SBATCH --time=48:00:00           
#SBATCH --job-name=PanCFPred
#SBATCH --output=/scratch/users/%u/%j.out
#
###################  END OF SLURM OPTIONS  ##################

module load cuda
source ~/miniconda3/etc/profile.d/conda.sh
cd /scratch_tmp/prj/inf_vr_audio_book/vraudiobook_final/vraudiobook

conda activate panfusion

WANDB_MODE=offline WANDB_NAME=PanimeCFPredict python main.py predict --data=PanimeDataModule --model=PanFusion --ckpt_path=./logs/a2tw2ymw/checkpoints/epoch_179.ckpt
