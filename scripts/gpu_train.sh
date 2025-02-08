#!/bin/bash -l
#
######################  SLURM OPTIONS  ######################
#SBATCH --partition=gpu              
#SBATCH --constraint=a100_80g       
#SBATCH --gres=gpu:1               
#SBATCH --nodes=1             
#SBATCH --cpus-per-gpu=4           
#SBATCH --mem=32G                  
#SBATCH --time=48:00:00           
#SBATCH --job-name=Panime
#SBATCH --output=/scratch/users/%u/%j.out
#
###################  END OF SLURM OPTIONS  ##################

module load cuda
source ~/miniconda3/etc/profile.d/conda.sh
cd /scratch_tmp/prj/inf_vr_audio_book/vraudiobook

conda activate panfusion

WANDB_NAME=Panime \
python main.py fit \
  --data=PanimeDataModule \
  --model=PanFusion \
  --trainer.max_epochs=200 \
  --data.batch_size=2 \
  --data.num_workers=1 \
  --model.ckpt_path=./logs/default_run/checkpoints/epoch_9.ckpt