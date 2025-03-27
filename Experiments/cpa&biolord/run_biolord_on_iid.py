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



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from scipy.stats import pearsonr

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
IID_EXPERIMENTS_PATH = "/fs/cbcb-scratch/cys/w1ot/iid_experiments"
OUTPUT_DIR = "/fs/cbcb-scratch/cys/w1ot/test_baselines/iid_experiments/biolord"
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

def get_perturbation_datasets():
    """Find datasets in the iid_experiments directory, limited to specific datasets."""
    # Only include these specific datasets
    dataset_dirs = ["sciplex3-hvg-top1k_50"]
    
    perturbation_paths = {}
    
    # For each dataset directory, find all perturbation subdirectories
    for dataset_dir in dataset_dirs:
        dataset_path = os.path.join(IID_EXPERIMENTS_PATH, dataset_dir)
        
        # Skip if the dataset directory doesn't exist
        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset directory {dataset_dir} not found in {IID_EXPERIMENTS_PATH}")
            continue
            
        perturbation_dirs = [d for d in os.listdir(dataset_path) 
                            if os.path.isdir(os.path.join(dataset_path, d))]
        
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
    elif "sciplex" in dataset_name:
        return "perturbation"
    else:
        # Default to "perturbation" for unknown datasets
        return "perturbation"

def extract_perturbation_name(pert_dir):
    """Extract the perturbation name from the directory name."""
    # Remove trailing _1, _2, etc. from the directory name (e.g., cisplatin_1 -> cisplatin)
    match = re.match(r"(.+)_\d+$", pert_dir)
    if match:
        return match.group(1)
    return pert_dir

def run_parameter_sweep():
    """
    Run a parameter sweep to find the best module parameters based on MMD score.
    Randomly selects one perturbation task to perform the search on.
    """
    print("Starting parameter sweep for bioLORD...", flush=True)
    
    # Set up environment
    setup()
    
    # Get all perturbation datasets
    perturbation_paths = get_perturbation_datasets()
    
    if not perturbation_paths:
        print("No perturbation datasets found in the iid_experiments directory.")
        return
    
    # Randomly select one perturbation task
    # seed random number generator
    random.seed(42)
    selected_key = random.choice(list(perturbation_paths.keys()))
    dataset_name, pert_dir = selected_key
    pert_path = perturbation_paths[selected_key]
    
    print(f"Randomly selected perturbation task: {dataset_name} - {pert_dir}")
    
    # Define parameter grid
    reduced_param_grid = {
        "decoder_width": [4096],
        "decoder_depth": [4],
        "attribute_nn_width": [2048],
        "attribute_nn_depth": [2],
        "n_latent_attribute_categorical": [3],
        "reconstruction_penalty": [0], # 5
        "unknown_attribute_penalty": [1], # 5
        "unknown_attribute_noise_param": [0.05], # 4 
        "attribute_dropout_rate": [0.1],
    }
    
    # Fixed parameters
    fixed_params = {
        "gene_likelihood": "normal",
        "use_batch_norm": False,
        "use_layer_norm": False,
        "seed": 42
    }
    
    # Determine perturbation attribute and name
    perturbation_attribute = determine_perturbation_attribute(dataset_name)
    perturbation_name = extract_perturbation_name(pert_dir)
    
    # Create output directory for sweep results
    sweep_output_dir = os.path.join(OUTPUT_DIR, "param_sweep", dataset_name, pert_dir)
    os.makedirs(sweep_output_dir, exist_ok=True)
    
    # Load data
    source_train = ad.read_h5ad(os.path.join(pert_path, "source_train.h5ad"))
    source_test = ad.read_h5ad(os.path.join(pert_path, "source_test.h5ad"))
    target_train = ad.read_h5ad(os.path.join(pert_path, "target_train.h5ad"))
    target_test = ad.read_h5ad(os.path.join(pert_path, "target_test.h5ad"))
    
    print(f"Data loaded:")
    print(f"  Source train: {source_train.shape}")
    print(f"  Source test: {source_test.shape}")
    print(f"  Target train: {target_train.shape}")
    print(f"  Target test: {target_test.shape}")
    
    # Log transform if needed
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
    
    # Ensure perturbation attribute is properly set
    if perturbation_attribute not in source_train.obs.columns:
        print(f"Adding {perturbation_attribute} column to AnnData objects")
        source_train.obs[perturbation_attribute] = 'control'
        source_test.obs[perturbation_attribute] = 'control'
        target_train.obs[perturbation_attribute] = perturbation_name
        target_test.obs[perturbation_attribute] = perturbation_name
    
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
    elif "sciplex" in dataset_name:
        LATENT_DIM = 50
    else:
        LATENT_DIM = 32  # Default value
    
    # Prepare for parameter sweep
    all_results = []
    
    # Use reduced grid for faster testing
    param_combinations = list(product(*reduced_param_grid.values()))
    total_combinations = len(param_combinations)
    print(f"Testing {total_combinations} parameter combinations")
    
    # Train shortened number of epochs for parameter sweep
    sweep_epochs = 100
    
    # Configure training parameters with better optimization settings
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
    
    # Run sweep
    for i, params in enumerate(param_combinations):
        # Create module params dictionary from current combination
        module_params = {k: v for k, v in zip(reduced_param_grid.keys(), params)}
        module_params.update(fixed_params)  # Add fixed parameters
        
        print(f"\nTesting parameter set {i+1}/{total_combinations}:")
        for k, v in module_params.items():
            print(f"  {k}: {v}")
        
        # Initialize the bioLORD model
        model_name = f"biolord_sweep_{i+1}"
        model = Biolord(
            adata=combined_adata,
            n_latent=LATENT_DIM,
            model_name=model_name,
            module_params=module_params,
            train_classifiers=False,
            split_key="split"
        )
        
        # Train the model with fewer epochs for the sweep
        print(f"Training bioLORD for {sweep_epochs} epochs...")
        model.train(
            max_epochs=sweep_epochs,
            batch_size=128,
            plan_kwargs=trainer_params,
            early_stopping=True,
            early_stopping_patience=50,  # Lower patience for faster sweeping
            check_val_every_n_epoch=1,
            accelerator="auto",
            device="auto",
        )
        
        print("Training completed.")
        
        # Generate counterfactual predictions
        adata_source = source_test.copy()
        target_attributes = [perturbation_attribute]
        
        adata_preds = model.compute_prediction_adata(
            combined_adata,
            adata_source,
            target_attributes=target_attributes,
            add_attributes=[]
        )
        
        perturbed_predictions = adata_preds[adata_preds.obs[perturbation_attribute] == perturbation_name].X
        perturbed_predictions = np.clip(perturbed_predictions, 0, np.inf)
        
        print("np.max(perturbed_predictions)", np.max(perturbed_predictions))
        print("np.min(perturbed_predictions)", np.min(perturbed_predictions))
        
        # Select genes for evaluation
        top_k = 50
        if top_k > target_test.shape[1]:
            gene_list = None
        else:
            try:
                # Try to load existing marker genes
                try:
                    perturbation = target_test.obs[perturbation_attribute].unique()[0]
                    gene_list = target_test.varm['marker_genes-drug-rank'][perturbation].sort_values()[:top_k].index
                except:
                    # If loading fails, compute differential expression
                    import scanpy as sc
                    combined_test = ad.concat([source_test, target_test])
                    sc.tl.rank_genes_groups(combined_test, perturbation_attribute, reference='control')
                    gene_list = combined_test.uns['rank_genes_groups']['names'][perturbation_name][:top_k]
            except Exception as e:
                print(f"Warning: Could not compute differential genes: {e}")
                gene_list = None
        
        # Calculate evaluation metrics
        transported_adata = ad.AnnData(X=perturbed_predictions, var=source_test.var)
        r2_score, l2norm, mmd_score = metrics(
            transported=transported_adata,
            target=target_test,
            gene_list=gene_list
        )
        
        # Store results
        param_result = {
            "model": "biolord",
            "param_set": i+1,
            "perturbation": perturbation_name,
            "cell_r2": r2_score,
            "cell_l2": l2norm,
            "cell_mmd": mmd_score,
        }
        # Add parameters to result
        for k, v in module_params.items():
            param_result[k] = v
        
        all_results.append(param_result)
        
        print(f"Parameter set {i+1} results:")
        print(f"  R2 score: {r2_score:.6f}")
        print(f"  L2 norm: {l2norm:.6f}")
        print(f"  MMD: {mmd_score:.6f}")
        
        # Save intermediate results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(sweep_output_dir, "param_sweep_results.csv"), index=False)
    
    # Find best parameters based on MMD score
    results_df = pd.DataFrame(all_results)
    best_params_idx = results_df['cell_mmd'].idxmin()
    best_params = results_df.iloc[best_params_idx].to_dict()
    
    print("\nParameter sweep completed!")
    print(f"Best parameters (lowest MMD score of {best_params['cell_mmd']:.6f}):")
    for k, v in best_params.items():
        if k in reduced_param_grid.keys():
            print(f"  {k}: {v}")
    
    # Save final results
    results_df.to_csv(os.path.join(sweep_output_dir, "param_sweep_results.csv"), index=False)
    
    # Save best parameters
    best_params_df = pd.DataFrame([best_params])
    best_params_df.to_csv(os.path.join(sweep_output_dir, "best_params.csv"), index=False)
    
    return best_params

def run_biolord_on_perturbation(dataset_name, pert_dir, pert_path, custom_module_params=None):
    """Run bioLORD on a specific perturbation dataset."""
    print(f"\nProcessing {dataset_name} - {pert_dir}")
    
    # Determine perturbation attribute (drug or perturbation)
    perturbation_attribute = determine_perturbation_attribute(dataset_name)
    perturbation_name = extract_perturbation_name(pert_dir)
    
    # Create output directory structure mirroring the input structure
    dataset_output_dir = os.path.join(OUTPUT_DIR, dataset_name, pert_dir)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # Load data
    source_train = ad.read_h5ad(os.path.join(pert_path, "source_train.h5ad"))
    source_test = ad.read_h5ad(os.path.join(pert_path, "source_test.h5ad"))
    target_train = ad.read_h5ad(os.path.join(pert_path, "target_train.h5ad"))
    target_test = ad.read_h5ad(os.path.join(pert_path, "target_test.h5ad"))

    # balance the number of cells in the source and target train sets

    num_cells = min(source_train.shape[0], target_train.shape[0])
    source_train = source_train[:num_cells]
    target_train = target_train[:num_cells]
    print("Balanced number of cells in source and target train sets", source_train.shape, target_train.shape)

    
    print(f"Data loaded:")
    print(f"  Source train: {source_train.shape}")
    print(f"  Source test: {source_test.shape}")
    print(f"  Target train: {target_train.shape}")
    print(f"  Target test: {target_test.shape}")

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
    
    # Ensure perturbation attribute is properly set
    # Set source cells to 'control' and target cells to the perturbation name
    if perturbation_attribute not in source_train.obs.columns:
        print(f"Adding {perturbation_attribute} column to AnnData objects")
        source_train.obs[perturbation_attribute] = 'control'
        source_test.obs[perturbation_attribute] = 'control'
        target_train.obs[perturbation_attribute] = perturbation_name
        target_test.obs[perturbation_attribute] = perturbation_name
    
    # Create a clean combined dataset for training
    combined_adata = ad.concat([source_train, target_train])
    
    # Add a split column to identify training and validation data
    combined_adata.obs['split'] = 'train'
    val_size = int(combined_adata.n_obs * 0.2)
    combined_adata.obs.iloc[-val_size:, combined_adata.obs.columns.get_loc('split')] = 'test'
    
    
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
    
    # Configure model parameters for better perturbation modeling
    if custom_module_params is None:
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
    else:
        # Use custom parameters if provided
        module_params = custom_module_params
        print("Using custom module parameters from parameter sweep")
    
    # Set latent dimension based on dataset
    if "4i" in dataset_name:
        LATENT_DIM = 8
    elif "sciplex" in dataset_name:
        LATENT_DIM = 50
    else:
        LATENT_DIM = 32  # Default value
    
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
    
    # Configure training parameters with better optimization settings
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
        device="auto",
    )
    
    print("Training completed.")
    
    # Evaluation using BioLORD's counterfactual prediction capabilities
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
    
    
    print("perturbation_name", perturbation_name)
    print(adata_preds.obs)
    perturbed_predictions = adata_preds[adata_preds.obs[perturbation_attribute] == perturbation_name].X
    print("perturbed_predictions.shape, adata_source.shape", perturbed_predictions.shape, adata_source.shape)
    perturbed_predictions = np.clip(perturbed_predictions, 0, np.inf)
    # Prepare data for evaluation - use only top 50 most differential genes
    # Select top 50 features for evaluation
    top_k = 50
    if top_k > target_test.shape[1]:
        top_k = target_test.shape[1]
        gene_list = None
        print("Using all genes for evaluation")
    else:
        try:
            # First try to load existing marker genes
            try:
                perturbation = target_test.obs[perturbation_attribute].unique()[0]
                gene_list = target_test.varm['marker_genes-drug-rank'][perturbation].sort_values()[:top_k].index
                print("Loaded existing gene list")
            except:
                # If loading fails, compute differential expression using scanpy
                print("Computing new differential gene list...")
                import scanpy as sc
                # Combine source and target test data
                combined_test = ad.concat([source_test, target_test])
                # Compute differential expression
                sc.tl.rank_genes_groups(combined_test, perturbation_attribute, reference='control')
                gene_list = combined_test.uns['rank_genes_groups']['names'][perturbation_name][:top_k]
                print(f"Created new gene list with top {top_k} differential genes")
        except Exception as e:
            print(f"Warning: Could not compute differential genes: {e}")
            gene_list = None
    
    # Create AnnData objects for evaluation metrics
    if gene_list is not None:
        print(f"Using top {len(gene_list)} differential genes for evaluation")
        transported_adata = ad.AnnData(X=perturbed_predictions, var=source_test.var)
        
        # Calculate evaluation metrics using the imported metrics function
        r2_score, l2norm, mmd_score = metrics(
            transported=transported_adata,
            target=target_test,
            gene_list=gene_list
        )
    else:
        print("Using all genes for evaluation")
        transported_adata = ad.AnnData(X=perturbed_predictions, var=source_test.var)
        
        # Calculate evaluation metrics using the imported metrics function
        r2_score, l2norm, mmd_score = metrics(
            transported=transported_adata,
            target=target_test
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
        "cell_r2": r2_score,
        "cell_l2": l2norm,
        "cell_mmd": mmd_score,
    }
    
    # Save results to the output directory
    results_file = os.path.join(dataset_output_dir, "results.csv")
    
    # Create new results dataframe
    results_df = pd.DataFrame([results])
    
    # Save results
    results_df.to_csv(results_file, index=False)
    
    print(f"Evaluation results:")
    print(f"  L2 norm: {l2norm:.6f}")
    print(f"  R2 score: {r2_score:.6f}")
    print(f"  MMD: {mmd_score:.6f}")
    print(f"Results saved to {results_file}")
    
    return results

def main():
    """Main function to run bioLORD on all perturbations."""
    # Force flushing of print statements to ensure log contains output
    import sys
    import time
    
    print("Starting bioLORD evaluation...", flush=True)
    print(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Python version: {sys.version}", flush=True)
    print(f"Working directory: {os.getcwd()}", flush=True)
    sys.stdout.flush()
    
    setup()
    
    # First run parameter sweep to find best module parameters
    print("Running parameter sweep to find optimal hyperparameters...")
    # best_params = run_parameter_sweep()
    
    # Get all perturbation datasets
    perturbation_paths = get_perturbation_datasets()
    
    if not perturbation_paths:
        print("No perturbation datasets found in the iid_experiments directory.")
        return
    
    print(f"Found {len(perturbation_paths)} perturbation datasets.")
    
    # Run bioLORD on each perturbation with best parameters
    results = []
    for (dataset_name, pert_dir), pert_path in perturbation_paths.items():
        # Use the best parameters found during the sweep
        pert_result = run_biolord_on_perturbation(dataset_name, pert_dir, pert_path, None)
        results.append(pert_result)
    
    # Summarize results
    print("\nEvaluation completed!")
    print(f"Processed {len(results)} out of {len(perturbation_paths)} perturbations successfully.")

if __name__ == "__main__":
    main()