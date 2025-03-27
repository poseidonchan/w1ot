#!/usr/bin/env python3

import os
import sys
import time
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm

# Add paths to import from both Wasserstein1Benchmark and w1ot
sys.path.append("/fs/cbcb-scratch/cys/w1ot/Wasserstein1Benchmark")
sys.path.append("/fs/cbcb-scratch/cys/w1ot")

# Import from Wasserstein1Benchmark for synthetic dataset generation and metrics
from src.map_benchmark import MixToOneBenchmark
from src.methods import L2_metrics, cosine_metrics
from src.methods import calculate_unbiased_wasserstein, calculate_kantorovitch_wasserstein

# Import from w1ot for the Sort-Out method testing
from w1ot.models import LBNN
from w1ot.ot import w1ot, OTDataset

# Configuration
GROUP_SIZES = [1, 2, 4, 8, 16]  # Group sizes for Sort-Out
FUNNEL_WIDTHS = [2, 8, 32, 128]  # Number of funnels in the mixture
DIMENSIONS = [32, 64, 128, 256]  # Test different dimensions
SEED = 42
BATCH_SIZE = 1024
NUM_ITERATIONS = 10000
NUM_SAMPLES = 100000
TEST_SAMPLES = 2000

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create results directory
results_dir = "/fs/cbcb-scratch/cys/w1ot/sort_out_results"
os.makedirs(results_dir, exist_ok=True)

# Prepare dataframe for results
results = []

def compute_metrics(ot_solver, benchmark, dim, width, group_size):
    """Compute metrics for the trained potential function using the benchmark metrics"""
    # Calculate metrics as per the original implementation
    w1_true = calculate_unbiased_wasserstein(benchmark, TEST_SAMPLES)
    w1_est = calculate_kantorovitch_wasserstein(ot_solver.phi, "sort_out", benchmark, TEST_SAMPLES)
    
    # Calculate relative deviation
    w1_rel_dev = abs(w1_true - w1_est) / abs(w1_true) if abs(w1_true) > 1e-6 else 0
    
    cosine_sim = cosine_metrics(ot_solver.phi, None, benchmark, TEST_SAMPLES, flag_rev=False)
    l2_dist = L2_metrics(ot_solver.phi, None, benchmark, TEST_SAMPLES, flag_rev=False)
    
    return {
        'dimension': dim,
        'width': width,  # Number of funnels
        'group_size': group_size,
        'w1_true': w1_true,
        'w1_est': w1_est,
        'w1_rel_dev': w1_rel_dev,
        'cosine_similarity': cosine_sim,
        'l2_distance': l2_dist
    }

def create_heatmaps(results_df):
    """Create and save heatmaps for each group size and metric"""
    metrics = ['cosine_similarity', 'l2_distance', 'w1_rel_dev']
    metric_titles = ['Cosine Similarity', 'L2 Distance', 'W1 Relative Deviation']
    
    # Create directory for heatmaps
    heatmap_dir = os.path.join(results_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # For each of the 5 group sizes, create 3 heatmaps (one for each metric)
    # This will result in 15 total heatmaps (5 group sizes x 3 metrics)
    for group_size in GROUP_SIZES:
        group_data = results_df[results_df['group_size'] == group_size]
        
        if len(group_data) == 0:
            continue
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
            # Create a pivot table for the heatmap
            pivot_data = group_data.pivot_table(
                values=metric,
                index='dimension',
                columns='width'
            )
            
            # Reorder indices to match desired order
            pivot_data = pivot_data.reindex(DIMENSIONS)
            pivot_data = pivot_data.reindex(FUNNEL_WIDTHS, axis=1)
            
            # Set colormap based on metric
            if metric == 'cosine_similarity':
                cmap = 'RdYlGn'  # Green is good (high cosine similarity)
                vmin = 0
                vmax = 1
            else:  # l2_distance, w1_rel_dev
                cmap = 'RdYlGn_r'  # Green is good (low distance/deviation)
                vmin = None
                vmax = None
            
            # Plot heatmap
            sns.heatmap(
                pivot_data, 
                annot=True, 
                fmt=".3f", 
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                ax=axes[i]
            )
            
            axes[i].set_title(f"{title} (Group Size = {group_size})")
            axes[i].set_xlabel("Number of Funnels")
            axes[i].set_ylabel("Dimension")
            
        plt.tight_layout()
        plt.savefig(os.path.join(heatmap_dir, f"heatmap_group_size_{group_size}.png"))
        plt.close()

def main():
    # Loop through each dimension, width, and group size
    for dim in DIMENSIONS:
        for width in FUNNEL_WIDTHS:
            for group_size in GROUP_SIZES:
                print(f"\n======== Testing dimension {dim}, funnels {width}, group size {group_size} ========")
                
                try:
                    benchmark = MixToOneBenchmark(dim=dim, width=width, seed=SEED, device=device)
                    print(f"Created benchmark with dimension {dim}, funnels {width}")
                except Exception as e:
                    print(f"Error creating benchmark: {e}")
                    continue

                # Generate training data
                X_train = benchmark.input_sampler.sample(NUM_SAMPLES).cpu().numpy()
                Y_train = benchmark.output_sampler.sample(NUM_SAMPLES).cpu().numpy()
                

                print(X_train.shape, Y_train.shape)
                # Model directory
                model_dir = os.path.join(results_dir, f"dim_{dim}_width_{width}_group_{group_size}")
                os.makedirs(model_dir, exist_ok=True)
                
                # Initialize w1ot solver
                ot_solver = w1ot(source=X_train, target=Y_train, device=device, path=model_dir)
                
                
                # Train the potential function with Sort-Out method
                start_time = time.time()
                ot_solver.fit_potential_function(
                    orthornormal_layer='cayley',  # Using Cayley for Sort-Out
                    groups=group_size,  # Set the group size for GroupSort activation
                    batch_size=BATCH_SIZE,
                    num_iters=NUM_ITERATIONS,
                )
                training_time = time.time() - start_time
                
                # Compute metrics
                metrics = compute_metrics(ot_solver, benchmark, dim, width, group_size)
                metrics['training_time'] = training_time
                
                # Store results
                results.append(metrics)
                
                # Save results to CSV after each run
                pd.DataFrame(results).to_csv(os.path.join(results_dir, "results.csv"), index=False)
                
                print(f"Completed testing for dimension {dim}, funnels {width}, group size {group_size}")
                print(f"True W1: {metrics['w1_true']:.4f}")
                print(f"Estimated W1: {metrics['w1_est']:.4f}")
                print(f"W1 Relative Deviation: {metrics['w1_rel_dev']:.4f}")
                print(f"Cosine similarity: {metrics['cosine_similarity']:.4f}")
                print(f"L2 distance: {metrics['l2_distance']:.4f}")
                print(f"Training time: {training_time:.2f} seconds")
    
    # Create heatmaps
    create_heatmaps(pd.DataFrame(results))
    print("\nAll testing completed. Results saved to CSV and heatmaps generated.")

if __name__ == "__main__":
    main()