#!/bin/bash
#SBATCH --job-name=biolord
#SBATCH --output=logs/run_biolord_on_iid.out
#SBATCH --error=logs/run_biolord_on_iid.err
#SBATCH --partition=cbcb-heng
#SBATCH --account=cbcb-heng
#SBATCH --qos=highmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1



source ~/.bashrc

conda activate biolord2

cd /fs/cbcb-scratch/cys/w1ot/


cd test_baselines

python -u run_biolord_on_geneperturb.py 2>&1 | tee biolord_geneperturb.log