#!/bin/sh
#SBATCH --job-name=chemcpa_geneperturb
#SBATCH --output=logs/run_chemcpa_on_geneperturb.out
#SBATCH --error=logs/run_chemcpa_on_geneperturb.err
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1


source ~/.bashrc

conda activate chemCPA

pip install scipy

cd /fs/cbcb-scratch/cys/w1ot/test_baselines

python -u run_chemcpa_on_geneperturb.py 2>&1 | tee chemcpa_geneperturb.log