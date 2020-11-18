#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=v100_sxm2_4
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-03:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=capstone
#SBATCH --mail-type=END
#SBATCH --output=slurm_%j.out


#command line argument
cd /scratch/tn709/capstone/Capstone
source setup.sh
export MPLBACKEND="pdf"
#conda activate cap_env
python -u modeling_v3.py 5000 100 100 0.001
