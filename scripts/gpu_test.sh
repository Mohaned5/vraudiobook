#!/bin/bash -l
#
######################  SLURM OPTIONS  ######################
#SBATCH --partition=gpu              
#SBATCH --gres=gpu:1               
#SBATCH --nodes=1             
#SBATCH --cpus-per-gpu=4           
#SBATCH --mem=32G                  
#SBATCH --time=48:00:00           
#SBATCH --job-name=PanimeTest
#SBATCH --output=/scratch/users/%u/panimetest.out
#
###################  END OF SLURM OPTIONS  ##################

module load cuda
source ~/miniconda3/etc/profile.d/conda.sh
cd /scratch_tmp/prj/inf_vr_audio_book/vraudiobook_final/vraudiobook

conda activate panfusion

WANDB_NAME=PanimeTest python main.py test --data=PanimeDataModule --model=PanFusion  --model.ckpt_path=./logs/a2tw2ymw/checkpoints/epoch_179.ckpt --data.num_workers=1