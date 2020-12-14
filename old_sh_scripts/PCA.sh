#!/bin/bash

##SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --job-name=pca
#SBATCH --mem=100GB
#SBATCH --mail-type=END
#SBATCH --output=slurm_%j.out
#SBATCH --time=7-00:00:00

python PCA.py