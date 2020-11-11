#!/bin/bash

#SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --job-name=modeling
#SBATCH --mem=100GB
#SBATCH --mail-type=END
#SBATCH --output=slurm_%j.out
#SBATCH --time=7-00:00:00

#command line argument
cd /scratch/fg746/capstone/Capstone
source setup.sh
#conda activate cap_env
python -u modeling.py

