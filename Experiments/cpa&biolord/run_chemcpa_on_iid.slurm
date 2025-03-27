#!/bin/sh
#SBATCH --job-name=chemcpa_iid
#SBATCH --output=logs/run_chemcpa_on_iid.out
#SBATCH --error=logs/run_chemcpa_on_iid.err
#SBATCH --partition=cbcb-heng
#SBATCH --account=cbcb-heng
#SBATCH --qos=highmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtxa5000:1


source ~/.bashrc

conda activate chemCPA

cd /fs/cbcb-scratch/cys/w1ot/test_baselines

python -u run_simple_chemcpa_on_iid.py 2>&1 | tee chemcpa_iid.log