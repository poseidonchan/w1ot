import os
import sys
import pandas as pd
import numpy as np
import anndata as ad
import torch
from pathlib import Path
import scanpy as sc
import glob
import re
import scipy.sparse
import importlib.util
import random
from itertools import product


from scipy.stats import pearsonr

def evaluate_perturbed_gene_logfc(
    predicted_adata,
    target_adata,
    perturbed_genes,
    expected_fold_change
) -> dict:
    """
    Evaluate the predicted log fold change of perturbed genes.
    
    Parameters:
    predicted_adata (AnnData): AnnData object with predicted gene expression
    target_adata (AnnData): AnnData object with actual target gene expression
    perturbed_genes (List[str]): List of gene names that were perturbed
    expected_fold_change (float): The fold change that was applied during perturbation
    
    Returns:
    dict: Dictionary with evaluation metrics for perturbed genes
    """
    # Calculate mean expression for perturbed genes
    predicted_means = []
    target_means = []
    
    for gene in perturbed_genes:
        gene_idx = np.where(predicted_adata.var_names == gene)[0][0]
        
        # Extract gene expression values
        if scipy.sparse.issparse(predicted_adata.X):
            pred_expr = predicted_adata.X[:, gene_idx].toarray().flatten()
            target_expr = target_adata.X[:, gene_idx].toarray().flatten()
        else:
            pred_expr = predicted_adata.X[:, gene_idx]
            target_expr = target_adata.X[:, gene_idx]
        
        # Calculate mean expression
        predicted_means.append(np.mean(pred_expr))
        target_means.append(np.mean(target_expr))
    
    # Convert to numpy arrays
    predicted_means = np.array(predicted_means)
    target_means = np.array(target_means)
    
    # Calculate log fold change error
    # Expected logFC is log2(expected_fold_change)
    expected_logfc = np.log2(expected_fold_change)
    
    # Calculate predicted logFC (comparing predictions to their actual mean)
    predicted_logfc = np.log2(predicted_means / np.mean(predicted_means))
    
    # Calculate target logFC (comparing target to their actual mean)
    target_logfc = np.log2(target_means / np.mean(target_means))
    
    # Calculate metrics
    logfc_error = np.mean(np.abs(predicted_logfc - target_logfc))
    logfc_corr = np.corrcoef(predicted_logfc, target_logfc)[0, 1]
    
    # Calculate accuracy (how close the predictions match the expected fold change)
    # For each perturbed gene, check if predicted logFC is within 25% of expected logFC
    correct_direction = np.sum((predicted_logfc > 0) == (target_logfc > 0))
    direction_accuracy = correct_direction / len(perturbed_genes)
    
    # Calculate R² for logFC
    ss_total = np.sum((target_logfc - np.mean(target_logfc))**2)
    ss_residual = np.sum((target_logfc - predicted_logfc)**2)
    logfc_r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
    
    return {
        'logfc_error': logfc_error,
        'logfc_corr': logfc_corr,
        'logfc_r2': logfc_r2,
        'direction_accuracy': direction_accuracy
    }

def ensure_numpy(data):
    """Convert data to numpy array."""
    if isinstance(data, np.ndarray):
        return data
    elif hasattr(data, 'toarray'):
        return data.toarray()
    else:
        return np.array(data)

def mmd_distance(x, y, gamma):
    """MMD distance with RBF kernel."""
    from sklearn.metrics.pairwise import rbf_kernel
        
    x = ensure_numpy(x)
    y = ensure_numpy(y)
        
    xx = rbf_kernel(x, x, gamma)
    xy = rbf_kernel(x, y, gamma)
    yy = rbf_kernel(y, y, gamma)
        
    return xx.mean() + yy.mean() - 2 * xy.mean()

def metrics(transported, target, gene_list=None, data_space='X'):
    """Calculate evaluation metrics between transported and target data."""
    if gene_list is not None:
        transported_data = ensure_numpy(transported[:, gene_list].X)
        target_data = ensure_numpy(target[:, gene_list].X)
    else:
        transported_data = ensure_numpy(transported.X)
        target_data = ensure_numpy(target.X)
            
    # Compare the feature means
    transported_mean = transported_data.mean(axis=0).flatten()
    target_mean = target_data.mean(axis=0).flatten()
        
    # Pearson correlation coefficient between means
    corr_coeff, _ = pearsonr(transported_mean, target_mean)
    r2 = corr_coeff ** 2
        
    # L2 distance between means
    l2_dist = np.linalg.norm(transported_mean - target_mean)
        
    # MMD distance
    gammas = np.logspace(1, -3, num=50)
    mmd_dist = np.mean([mmd_distance(target_data, transported_data, g) for g in gammas])
        
    return r2, l2_dist, mmd_dist

# Constants
GENE_PERTURBATION_PATH = "/fs/cbcb-scratch/cys/w1ot/gene_perturbation_experiments"
OUTPUT_DIR = "/fs/cbcb-scratch/cys/w1ot/test_baselines/gene_perturbation_experiments/biolord"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 500

def normalize_and_log_transform(X):
    """Normalize and log transform data."""
    X = np.array(X)
    X = np.log1p(X)
    return X

def setup():
    """Create output directory and prepare for testing."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Setup complete. Output will be saved to {OUTPUT_DIR}", flush=True)
    print(f"Using device: {DEVICE}", flush=True)

    try:
        from biolord import Biolord
        print("Successfully imported Biolord class", flush=True)
    except Exception as e:
        print(f"Error importing Biolord class: {e}", flush=True)

def get_gene_perturbation_datasets():
    """Find datasets in the gene_perturbation_experiments directory."""
    perturbation_paths = {}
    
    if not os.path.exists(GENE_PERTURBATION_PATH):
        print(f"Warning: Gene perturbation directory {GENE_PERTURBATION_PATH} not found")
        return perturbation_paths
    
    dataset_dirs = [d for d in os.listdir(GENE_PERTURBATION_PATH) 
                  if os.path.isdir(os.path.join(GENE_PERTURBATION_PATH, d))]
    
    # For each dataset directory, find all perturbation subdirectories
    for dataset_dir in dataset_dirs:
        dataset_path = os.path.join(GENE_PERTURBATION_PATH, dataset_dir)
        
        perturbation_dirs = [d for d in os.listdir(dataset_path) 
                           if os.path.isdir(os.path.join(dataset_path, d)) and 'gene_perturb' in d]
        
        # Store paths to each perturbation's data
        for pert_dir in perturbation_dirs:
            pert_path = os.path.join(dataset_path, pert_dir)
            
            # Check if this directory has the required data files
            if (os.path.exists(os.path.join(pert_path, "source_train.h5ad")) and
                os.path.exists(os.path.join(pert_path, "source_test.h5ad")) and
                os.path.exists(os.path.join(pert_path, "target_train.h5ad")) and
                os.path.exists(os.path.join(pert_path, "target_test.h5ad"))):
                
                perturbation_paths[(dataset_dir, pert_dir)] = pert_path
    
    return perturbation_paths

def determine_perturbation_attribute(dataset_name):
    """Determine the perturbation attribute based on the dataset name."""
    if "4i" in dataset_name:
        return "drug"
    else:
        # Default to "perturbation" for unknown datasets
        return "perturbation"

def run_biolord_on_perturbation(dataset_name, pert_dir, pert_path):
    """Run bioLORD on a specific gene perturbation dataset."""
    print(f"\nProcessing {dataset_name} - {pert_dir}")
    
    # Determine perturbation attribute
    perturbation_attribute = determine_perturbation_attribute(dataset_name)
    
    # Create output directory structure mirroring the input structure
    dataset_output_dir = os.path.join(OUTPUT_DIR, dataset_name, pert_dir)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # Load data
    source_train = ad.read_h5ad(os.path.join(pert_path, "source_train.h5ad"))
    source_test = ad.read_h5ad(os.path.join(pert_path, "source_test.h5ad"))
    target_train = ad.read_h5ad(os.path.join(pert_path, "target_train.h5ad"))
    target_test = ad.read_h5ad(os.path.join(pert_path, "target_test.h5ad"))
    
    # Load perturbed genes list
    perturbed_genes = np.load(os.path.join(pert_path, "perturbed_genes.npy"), allow_pickle=True)
    
    # Extract fold change from metadata if available
    fold_change = 2.0  # Default value
    if 'gene_perturbation' in target_train.uns and 'fold_change' in target_train.uns['gene_perturbation']:
        fold_change = target_train.uns['gene_perturbation']['fold_change']
    
    perturbation_name = target_train.obs[perturbation_attribute].unique()[0]
    
    print(f"Data loaded:")
    print(f"  Source train: {source_train.shape}")
    print(f"  Source test: {source_test.shape}")
    print(f"  Target train: {target_train.shape}")
    print(f"  Target test: {target_test.shape}")
    print(f"  Perturbed genes: {len(perturbed_genes)}")
    print(f"  Fold change: {fold_change}")

    # Balance the number of cells in the source and target train sets
    num_cells = min(source_train.shape[0], target_train.shape[0])
    source_train = source_train[:num_cells]
    target_train = target_train[:num_cells]
    print("Balanced number of cells in source and target train sets", source_train.shape, target_train.shape)
    
    # Apply log transformation if needed
    if source_train.X.max() > 50:
        print("Log transforming source train data", source_train.X.max())
        source_train.X = normalize_and_log_transform(ensure_numpy(source_train.X))
    
    if source_test.X.max() > 50:
        print("Log transforming source test data", source_test.X.max())
        source_test.X = normalize_and_log_transform(ensure_numpy(source_test.X))
    
    if target_train.X.max() > 50:
        print("Log transforming target train data", target_train.X.max())
        target_train.X = normalize_and_log_transform(ensure_numpy(target_train.X))
    
    if target_test.X.max() > 50:
        print("Log transforming target test data", target_test.X.max())
        target_test.X = normalize_and_log_transform(ensure_numpy(target_test.X))
    
    # Create a clean combined dataset for training
    combined_adata = ad.concat([source_train, target_train])
    
    # Add a split column to identify training and validation data
    combined_adata.obs['split'] = 'train'
    val_size = int(combined_adata.n_obs * 0.2)
    combined_adata.obs.iloc[-val_size:, combined_adata.obs.columns.get_loc('split')] = 'test'
    
    # Critical step: Add _indices column before setup_anndata
    combined_adata.obs['_indices'] = np.arange(combined_adata.n_obs)
    source_test.obs['_indices'] = np.arange(source_test.n_obs)
    target_test.obs['_indices'] = np.arange(target_test.n_obs)
    
    # Set up anndata for bioLORD
    print("Setting up AnnData for bioLORD...")
    from biolord import Biolord
    Biolord.setup_anndata(
        combined_adata,
        categorical_attributes_keys=[perturbation_attribute],
        ordered_attributes_keys=[],
    )
    
    # Set latent dimension based on dataset
    if "4i" in dataset_name:
        LATENT_DIM = 8
    else:
        LATENT_DIM = 50
    
    # Configure model parameters
    module_params = {
        "decoder_width": 4096,
        "decoder_depth": 4,
        "attribute_nn_width": 2048,
        "attribute_nn_depth": 2,
        "n_latent_attribute_categorical": 3,
        "gene_likelihood": "normal",
        "reconstruction_penalty": 100,
        "unknown_attribute_penalty": 100,
        "unknown_attribute_noise_param": 0.01,
        "attribute_dropout_rate": 0.1,
        "use_batch_norm": False,
        "use_layer_norm": False,
        "seed": 42
    }
    
    print(f"Using latent dimension: {LATENT_DIM} for dataset: {dataset_name}")
    
    # Initialize the bioLORD model
    print("Initializing bioLORD model...")
    model = Biolord(
        adata=combined_adata,
        n_latent=LATENT_DIM,
        model_name=f"biolord_{dataset_name}_{pert_dir}",
        module_params=module_params,
        train_classifiers=False,
        split_key="split"
    )
    
    # Configure training parameters
    trainer_params = {
        "latent_lr": 1e-4,
        "latent_wd": 1e-4,
        "decoder_lr": 1e-4,
        "decoder_wd": 1e-4,
        "attribute_nn_lr": 1e-2,
        "attribute_nn_wd": 4e-8,
        "cosine_scheduler": True,
        "scheduler_final_lr": 1e-5,
    }
    
    # Train the model
    print(f"Training bioLORD for {NUM_EPOCHS} epochs...")
    model.train(
        max_epochs=NUM_EPOCHS,
        batch_size=128,
        plan_kwargs=trainer_params,
        early_stopping=True,
        early_stopping_patience=50,
        check_val_every_n_epoch=1,
        accelerator="auto",
        monitor="val_biolord_metric",
        enable_checkpointing=False,
    )
    
    print("Training completed.")
    
    # Generate counterfactual predictions
    print("Generating counterfactual predictions...")
    
    # Use source_test (control cells) as the source for counterfactual predictions
    adata_source = source_test.copy()
    
    # Generate counterfactual predictions using bioLORD
    target_attributes = [perturbation_attribute]
    
    adata_preds = model.compute_prediction_adata(
        combined_adata,  # Reference for attribute categories
        adata_source,    # Source cells (control)
        target_attributes=target_attributes,  # Target attribute to modify
        add_attributes=[]  # Additional attributes to copy
    )
    
    perturbed_predictions = adata_preds[adata_preds.obs[perturbation_attribute] == perturbation_name].X
    perturbed_predictions = np.clip(perturbed_predictions, 0, np.inf)
    
    # Create AnnData object for evaluation
    transported_adata = ad.AnnData(X=perturbed_predictions, var=source_test.var)
    
    # Evaluate log fold change on perturbed genes
    print(f"Evaluating log fold change for {len(perturbed_genes)} perturbed genes...")
    logfc_metrics = evaluate_perturbed_gene_logfc(
        transported_adata,
        target_test,
        perturbed_genes,
        fold_change
    )
    
    # Save the predicted AnnData object
    predictions_dir = os.path.join(dataset_output_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    transported_adata.write(os.path.join(predictions_dir, 'biolord_predicted.h5ad'))
    print(f"Saved prediction to {os.path.join(predictions_dir, 'biolord_predicted.h5ad')}")
    
    # Create result dictionary
    results = {
        "model": "biolord",
        "perturbation": perturbation_name,
        "latent_dim": LATENT_DIM,
        "logfc_error": logfc_metrics['logfc_error'],
        "logfc_corr": logfc_metrics['logfc_corr'],
        "logfc_r2": logfc_metrics['logfc_r2'],
        "direction_accuracy": logfc_metrics['direction_accuracy'],
        "num_perturbed_genes": len(perturbed_genes),
        "fold_change": fold_change
    }
    
    # Save results to the output directory
    results_file = os.path.join(dataset_output_dir, "results.csv")
    
    # Create new results dataframe
    results_df = pd.DataFrame([results])
    
    # Save results
    results_df.to_csv(results_file, index=False)
    
    print(f"Evaluation results:")
    print(f"  LogFC error: {logfc_metrics['logfc_error']:.6f}")
    print(f"  LogFC correlation: {logfc_metrics['logfc_corr']:.6f}")
    print(f"  LogFC R²: {logfc_metrics['logfc_r2']:.6f}")
    print(f"  Direction accuracy: {logfc_metrics['direction_accuracy']:.6f}")
    print(f"Results saved to {results_file}")
    
    return results

def main():
    """Main function to run bioLORD on gene perturbation datasets."""
    # Force flushing of print statements to ensure log contains output
    import sys
    import time
    
    print("Starting bioLORD evaluation on gene perturbation data...", flush=True)
    print(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Python version: {sys.version}", flush=True)
    print(f"Working directory: {os.getcwd()}", flush=True)
    sys.stdout.flush()
    
    setup()
    
    # Get all gene perturbation datasets
    perturbation_paths = get_gene_perturbation_datasets()
    
    if not perturbation_paths:
        print("No gene perturbation datasets found in the directory.")
        return
    
    print(f"Found {len(perturbation_paths)} gene perturbation datasets.")
    
    # Run bioLORD on each perturbation
    results = []
    for (dataset_name, pert_dir), pert_path in perturbation_paths.items():
        pert_result = run_biolord_on_perturbation(dataset_name, pert_dir, pert_path)
        results.append(pert_result)
    
    # Summarize results
    print("\nEvaluation completed!")
    print(f"Processed {len(results)} out of {len(perturbation_paths)} perturbations successfully.")

if __name__ == "__main__":
    main() 