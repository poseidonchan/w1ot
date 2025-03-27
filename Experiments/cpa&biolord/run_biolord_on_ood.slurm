#!/bin/bash
#SBATCH --job-name=biolord_ood
#SBATCH --nodelist=cbcb28
#SBATCH --output=logs/run_biolord_on_ood.out
#SBATCH --error=logs/run_biolord_on_ood.err
#SBATCH --partition=cbcb-heng
#SBATCH --account=cbcb-heng
#SBATCH --qos=highmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1


# Activate conda environment
source ~/.bashrc

# activate biolord environment
# conda create -n biolord2 python=3.9
conda activate biolord2

# pip install biolord
# pip install scanpy

cd /fs/cbcb-scratch/cys/w1ot/


cd test_baselines
# Run with output to both the SLURM log and a separate log file
python -u run_biolord_on_ood.py 2>&1 | tee biolord_ood.log 