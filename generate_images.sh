#!/bin/bash

#SBATCH --cpus-per-task=1
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=simulator
#SBATCH --mem=100GB
#SBATCH --mail-type=END
#SBATCH --output=slurm_%j.out
#SBATCH --time=7-00:00:00

#command line argument
cd /scratch/tn709/capstone/Sim2600/sim2600
#source setup.sh
#conda activate cap_env
python nn_generate_ims.py
