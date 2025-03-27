import numpy as np
import anndata as ad
from anndata import AnnData
import scanpy as sc
import os
import pandas as pd
import ray
import random
import logging
from typing import List
import scipy.sparse

from w1ot.experiments import PerturbModel
from w1ot.utils import normalize_and_log_transform, ensure_numpy


def evaluate_perturbed_gene_logfc(
    predicted_adata: AnnData,
    target_adata: AnnData,
    perturbed_genes: List[str],
    expected_fold_change: float
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


@ray.remote(
    num_cpus=2,
    num_gpus=1,
    max_retries=-1,
    memory=1.6e+10,
)
def run_gene_perturbation_task(
    adata: AnnData,
    datasets_name: str,
    output_dir: str,
    percent_expressed: float,
    num_genes: int,
    fold_change: float,
    perturbation_attribute: str,
    test_size: float,
    device: str,
    embedding: bool,
    latent_dims: List[int],
    num_run: int,
    start_run: int,
    run_id: int
):
    """
    Run a gene perturbation task.

    Parameters:
    adata (AnnData): AnnData object containing the dataset.
    datasets_name (str): Name of the dataset.
    output_dir (str): Directory to save outputs.
    percent_expressed (float): Minimum percentage of cells where genes should be expressed.
    num_genes (int): Number of genes to perturb.
    fold_change (float): Fold change to apply to perturbed genes.
    perturbation_attribute (str): Attribute for perturbation.
    test_size (float): Proportion of the dataset to include in the test split.
    device (str): Device to be used for computation.
    embedding (bool): Whether to use embedding.
    latent_dims (List[int]): List of latent dimensions to consider.
    num_run (int): Number of runs.
    start_run (int): Starting run number.
    run_id (int): Unique identifier for this run.
    """
    import os
    import pandas as pd
    import numpy as np
    import anndata as ad
    import scipy.sparse
    
    perturbation = f"gene_perturb_{run_id}"
    print(f"Running gene perturbation task for run {run_id}")

    for run in range(1, num_run + 1):
        run_output_dir = os.path.join(
            output_dir, datasets_name, f"{perturbation}_{run + start_run}"
        )
        if not os.path.exists(run_output_dir):
            os.makedirs(run_output_dir)

        # Check if data split exists
        if all(os.path.exists(os.path.join(run_output_dir, f'{split}.h5ad'))
               for split in ['source_train', 'source_test', 'target_train', 'target_test']):
            print(f"Loading existing data split for {perturbation}, run {run}")
            source_train_adata = ad.read_h5ad(os.path.join(run_output_dir, 'source_train.h5ad'))
            source_test_adata = ad.read_h5ad(os.path.join(run_output_dir, 'source_test.h5ad'))
            target_train_adata = ad.read_h5ad(os.path.join(run_output_dir, 'target_train.h5ad'))
            target_test_adata = ad.read_h5ad(os.path.join(run_output_dir, 'target_test.h5ad'))
            
            # Load the perturbed gene list
            perturbed_genes = np.load(os.path.join(run_output_dir, 'perturbed_genes.npy'), allow_pickle=True)
            
        else:
            print(f"Creating new data split for {perturbation}, run {run}")
            
            # 1. Get the control set in the anndata
            control_adata = adata[adata.obs[perturbation_attribute] == 'control'].copy()
            
            # Randomly shuffle the data
            control_adata = control_adata[np.random.permutation(len(control_adata))]
            
            # 2. Find genes that are expressed in at least {percent_expressed} cells
            X = ensure_numpy(control_adata.X)
            gene_expression_rate = np.sum(X > 0, axis=0) / X.shape[0]
            candidate_genes = np.where(gene_expression_rate >= percent_expressed)[0]
            
            print(f"Found {len(candidate_genes)} candidate genes expressed in at least {percent_expressed*100}% of cells")
            
            if len(candidate_genes) < num_genes:
                print(f"Warning: Only {len(candidate_genes)} genes meet the expression criteria, using all of them.")
                num_genes_to_perturb = len(candidate_genes)
            else:
                num_genes_to_perturb = num_genes
            
            # 3. Randomly select genes to perturb
            np.random.seed(run + start_run)
            perturbed_gene_indices = np.random.choice(candidate_genes, size=num_genes_to_perturb, replace=False)
            perturbed_genes = control_adata.var_names[perturbed_gene_indices].tolist()
            
            print(f"Selected {len(perturbed_genes)} genes to perturb: {perturbed_genes[:5]}...")
            
            # Save the list of perturbed genes
            np.save(os.path.join(run_output_dir, 'perturbed_genes.npy'), perturbed_genes)
            
            # 4. Create perturbed data by increasing expression of selected genes
            # Split control data into train and test sets
            test_size_actual = max(500, min(1000, int(len(control_adata) * test_size)))
            
            source_train_adata = control_adata[:-test_size_actual].copy()
            source_test_adata = control_adata[-test_size_actual:].copy()
            
            # Create target data by perturbing gene expression
            target_train_adata = source_train_adata.copy()
            target_test_adata = source_test_adata.copy()
            
            # Apply fold change to selected genes
            for idx, gene in enumerate(perturbed_genes):
                gene_idx = np.where(target_train_adata.var_names == gene)[0][0]
                
                # Apply perturbation to training data
                if scipy.sparse.issparse(target_train_adata.X):
                    X_train = target_train_adata.X.toarray()
                    X_train[:, gene_idx] = X_train[:, gene_idx] * fold_change
                    target_train_adata.X = scipy.sparse.csr_matrix(X_train)
                else:
                    target_train_adata.X[:, gene_idx] = target_train_adata.X[:, gene_idx] * fold_change
                
                # Apply perturbation to test data
                if scipy.sparse.issparse(target_test_adata.X):
                    X_test = target_test_adata.X.toarray()
                    X_test[:, gene_idx] = X_test[:, gene_idx] * fold_change
                    target_test_adata.X = scipy.sparse.csr_matrix(X_test)
                else:
                    target_test_adata.X[:, gene_idx] = target_test_adata.X[:, gene_idx] * fold_change
            
            # Add perturbation attribute
            for adata_obj, pert_value in [(source_train_adata, 'control'), 
                                          (source_test_adata, 'control'),
                                          (target_train_adata, perturbation),
                                          (target_test_adata, perturbation)]:
                adata_obj.obs[perturbation_attribute] = pert_value
                
                # Store metadata about the perturbation
                adata_obj.uns['gene_perturbation'] = {
                    'perturbed_genes': perturbed_genes,
                    'fold_change': fold_change,
                    'percent_expressed': percent_expressed
                }
            
            # Save the split data
            source_train_adata.write(os.path.join(run_output_dir, 'source_train.h5ad'))
            source_test_adata.write(os.path.join(run_output_dir, 'source_test.h5ad'))
            target_train_adata.write(os.path.join(run_output_dir, 'target_train.h5ad'))
            target_test_adata.write(os.path.join(run_output_dir, 'target_test.h5ad'))

        # List to store results for this run
        run_results = []

        for latent_dim in latent_dims:
            for model_name in [ 'scgen', 'w1ot', 'w2ot']:

                model_output_dir = os.path.join(
                    run_output_dir,
                    f'saved_models',
                    model_name
                )
                if not os.path.exists(model_output_dir):
                    os.makedirs(model_output_dir)

                # Initialize and train the model
                if '4i' in datasets_name:
                    if model_name == 'scgen':
                        pmodel = PerturbModel(
                            model_name=model_name,
                            source_adata=source_train_adata,
                            target_adata=target_train_adata,
                            perturbation_attribute=perturbation_attribute,
                            latent_dim=8,
                            embedding=True,
                            output_dir=model_output_dir,
                            hidden_layers=[32, 32],
                            num_iters=10000,
                            device=device
                        )

                        pmodel.train()

                        # Get predictions
                        predicted_adata = pmodel.predict(source_test_adata)
                        
                        # Evaluate only the log fold change of perturbed genes
                        logfc_metrics = evaluate_perturbed_gene_logfc(
                            predicted_adata, 
                            target_test_adata, 
                            perturbed_genes, 
                            fold_change
                        )

                    else:
                        pmodel = PerturbModel(
                            model_name=model_name,
                            source_adata=source_train_adata,
                            target_adata=target_train_adata,
                            perturbation_attribute=perturbation_attribute,
                            latent_dim=latent_dim,
                            embedding=False,  # w1ot and w2ot don't need embedding for 4i
                            output_dir=model_output_dir,
                            device=device
                        )

                        pmodel.train()

                        # Get predictions
                        predicted_adata = pmodel.predict(source_test_adata)
                        
                        # Evaluate only the log fold change of perturbed genes
                        logfc_metrics = evaluate_perturbed_gene_logfc(
                            predicted_adata, 
                            target_test_adata, 
                            perturbed_genes, 
                            fold_change
                        )
                
                elif datasets_name in ['sciplex3-hvg-top1k']:
                    if model_name == 'scgen':
                        pmodel = PerturbModel(
                            model_name=model_name,
                            source_adata=source_train_adata,
                            target_adata=target_train_adata,
                            perturbation_attribute=perturbation_attribute,
                            latent_dim=50,
                            embedding=True,
                            output_dir=model_output_dir,
                            hidden_layers=[512, 512],
                            num_iters=250000,
                            device=device
                        )

                        pmodel.train()

                        # Get predictions
                        predicted_adata = pmodel.predict(source_test_adata)
                        
                        # Evaluate only the log fold change of perturbed genes
                        logfc_metrics = evaluate_perturbed_gene_logfc(
                            predicted_adata, 
                            target_test_adata, 
                            perturbed_genes, 
                            fold_change
                        )

                    else:
                        pmodel = PerturbModel(
                            model_name=model_name,
                            source_adata=source_train_adata,
                            target_adata=target_train_adata,
                            perturbation_attribute=perturbation_attribute,
                            latent_dim=latent_dim,
                            embedding=False,
                            output_dir=model_output_dir,
                            device=device
                        )

                        pmodel.train()

                        # Get predictions
                        predicted_adata = pmodel.predict(source_test_adata)
                        
                        # Evaluate only the log fold change of perturbed genes
                        logfc_metrics = evaluate_perturbed_gene_logfc(
                            predicted_adata, 
                            target_test_adata, 
                            perturbed_genes, 
                            fold_change
                        )

                else:
                    if model_name == 'scgen':
                        pmodel = PerturbModel(
                            model_name=model_name,
                            source_adata=source_train_adata,
                            target_adata=target_train_adata,
                            perturbation_attribute=perturbation_attribute,
                            latent_dim=50,
                            embedding=True,
                            output_dir=model_output_dir,
                            hidden_layers=[512, 512],
                            num_iters=250000,
                            device=device
                        )

                        pmodel.train()

                        # Get predictions
                        predicted_adata = pmodel.predict(source_test_adata)
                        
                        # Evaluate only the log fold change of perturbed genes
                        logfc_metrics = evaluate_perturbed_gene_logfc(
                            predicted_adata, 
                            target_test_adata, 
                            perturbed_genes, 
                            fold_change
                        )
                    
                    else:  # w1ot or w2ot for standard datasets
                        pmodel = PerturbModel(
                            model_name=model_name,
                            source_adata=source_train_adata,
                            target_adata=target_train_adata,
                            perturbation_attribute=perturbation_attribute,
                            latent_dim=latent_dim,
                            embedding=embedding,
                            output_dir=model_output_dir,
                            device=device
                        )

                        pmodel.train()

                        # Get predictions
                        predicted_adata = pmodel.predict(source_test_adata)
                        
                        # Evaluate only the log fold change of perturbed genes
                        logfc_metrics = evaluate_perturbed_gene_logfc(
                            predicted_adata, 
                            target_test_adata, 
                            perturbed_genes, 
                            fold_change
                        )

                # Save the results
                model_results = {
                    'model': model_name,
                    'perturbation': perturbation,
                    'latent_dim': latent_dim,
                    'logfc_error': logfc_metrics['logfc_error'],
                    'logfc_corr': logfc_metrics['logfc_corr'],
                    'logfc_r2': logfc_metrics['logfc_r2'],
                    'direction_accuracy': logfc_metrics['direction_accuracy'],
                    'num_perturbed_genes': len(perturbed_genes),
                    'perturbed_genes': ','.join(perturbed_genes[:10]) + '...' if len(perturbed_genes) > 10 else ','.join(perturbed_genes),
                    'fold_change': fold_change
                }
                run_results.append(model_results)

        # Save the run results to a CSV file
        results_df = pd.DataFrame(run_results)
        results_df.to_csv(os.path.join(run_output_dir, 'results.csv'), index=False)
    
    print(f"Finished running gene perturbation task for run {run_id}")


def run_gene_perturbation(
    dataset_path: str,
    output_dir: str = './gene_perturbation_experiments/',
    perturbation_attribute: str = 'drug',
    percent_expressed: float = 0.1,
    num_genes: int = 50,
    fold_change: float = 2.0,
    test_size: float = 0.2,
    device: str = 'cuda',
    embedding: bool = True,
    latent_dims: List[int] = [50],
    num_run: int = 1,
    start_run: int = 0,
    num_experiments: int = 5
) -> List:
    """
    Run gene perturbation experiments.

    Parameters:
    dataset_path (str): Path to the dataset file.
    output_dir (str): Directory to save outputs. Default is './gene_perturbation_experiments/'.
    perturbation_attribute (str): Attribute for perturbation. Default is 'perturbation'.
    percent_expressed (float): Minimum percentage of cells where genes should be expressed. Default is 0.1 (10%).
    num_genes (int): Number of genes to perturb. Default is 50.
    fold_change (float): Fold change to apply to perturbed genes. Default is 2.0.
    test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
    device (str): Device to be used for computation. Default is 'cuda'.
    embedding (bool): Whether to use embedding. Default is True.
    latent_dims (List[int]): List of latent dimensions to use. Default is [50].
    num_run (int): Number of runs to perform per experiment. Default is 1.
    start_run (int): Starting run index. Default is 0.
    num_experiments (int): Number of different gene perturbation experiments to run. Default is 5.

    Returns:
    List: List of tasks submitted to Ray.
    """
    import os
    import pandas as pd
    import numpy as np
    import anndata as ad

    datasets_name = os.path.splitext(os.path.basename(dataset_path))[0]

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the dataset once and put it into Ray's object store
    adata = ad.read_h5ad(dataset_path)
    adata_ref = ray.put(adata)

    # Submit tasks to Ray for each gene perturbation experiment
    tasks = []
    for exp_id in range(1, num_experiments + 1):
        print(f"Submitting gene perturbation task {exp_id}/{num_experiments}")
        task = run_gene_perturbation_task.remote(
            adata_ref,
            datasets_name,
            output_dir,
            percent_expressed,
            num_genes,
            fold_change,
            perturbation_attribute,
            test_size,
            device,
            embedding,
            latent_dims,
            num_run,
            start_run,
            exp_id
        )
        tasks.append(task)

    return tasks


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run gene perturbation experiments')
    parser.add_argument('--dataset', type=str, default="/fs/cbcb-scratch/cys/w1ot/datasets/4i-melanoma-8h-48.h5ad", help='Path to dataset file')
    parser.add_argument('--output_dir', type=str, default='/fs/cbcb-scratch/cys/w1ot/gene_perturbation_experiments/', help='Output directory')
    parser.add_argument('--perturbation_attribute', type=str, default='drug', help='Perturbation attribute')
    parser.add_argument('--percent_expressed', type=float, default=0.6, help='Minimum percentage of cells where genes should be expressed')
    parser.add_argument('--num_genes', type=int, default=5, help='Number of genes to perturb')
    parser.add_argument('--fold_change', type=float, default=2.0, help='Fold change to apply to perturbed genes')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--embedding', action='store_true', help='Use embedding')
    parser.add_argument('--latent_dim', type=int, default=50, help='Latent dimension')
    parser.add_argument('--num_run', type=int, default=1, help='Number of runs per experiment')
    parser.add_argument('--start_run', type=int, default=0, help='Starting run number')
    parser.add_argument('--num_experiments', type=int, default=5, help='Number of different gene perturbation experiments')
    
    args = parser.parse_args()
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()
    
    tasks = run_gene_perturbation(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        perturbation_attribute=args.perturbation_attribute,
        percent_expressed=args.percent_expressed,
        num_genes=args.num_genes,
        fold_change=args.fold_change,
        test_size=args.test_size,
        device=args.device,
        embedding=args.embedding,
        latent_dims=[args.latent_dim],
        num_run=args.num_run,
        start_run=args.start_run,
        num_experiments=args.num_experiments
    )
    
    # Wait for all tasks to complete
    ray.get(tasks)
    print("All gene perturbation experiments completed.") 