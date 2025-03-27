import os
import sys
import pandas as pd
import numpy as np
import anndata as ad
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import scanpy as sc
import re
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import random
from itertools import product


from scipy.stats import pearsonr
import scipy
from chemCPA.model import MLP  # Reuse the MLP implementation from chemCPA


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


# Simplified ChemCPA model for gene perturbation experiments
class SimplifiedChemCPA(nn.Module):
    def __init__(self, num_genes, num_conditions, latent_dim=32, 
                 encoder_width=256, encoder_depth=4, 
                 decoder_width=256, decoder_depth=4,
                 adversary_width=128, adversary_depth=3,
                 reg_adversary=9.100951626404369, penalty_adversary=0.4550475813202185,
                 dropout=0.262378,
                 device="cuda"):
        super(SimplifiedChemCPA, self).__init__()
        
        self.num_genes = num_genes
        self.num_conditions = num_conditions
        self.latent_dim = latent_dim
        self.reg_adversary = reg_adversary
        self.penalty_adversary = penalty_adversary
        self.device = device
        self.dropout = dropout  # Store dropout but don't use it directly in MLP (not supported)
        
        # Define encoder network (no dropout - not supported by MLP)
        self.encoder = MLP(
            [num_genes] + [encoder_width] * encoder_depth + [latent_dim], 
            last_layer_act="linear",
            batch_norm=True
        )
        
        # Define decoder network (no dropout - not supported by MLP)
        self.decoder = MLP(
            [latent_dim] + [decoder_width] * decoder_depth + [num_genes*2],  # *2 for mean and variance
            last_layer_act="linear", 
            batch_norm=True
        )
        
        # Define condition adversary (no dropout - not supported by MLP)
        self.adversary = MLP(
            [latent_dim] + [adversary_width] * adversary_depth + [num_conditions],
            last_layer_act="linear",
            batch_norm=True
        )
        
        # Define condition embeddings
        self.condition_embeddings = nn.Embedding(num_conditions, latent_dim)
        
        # Define loss functions
        self.loss_reconstruction = nn.GaussianNLLLoss()
        self.loss_adversary = nn.CrossEntropyLoss()
        
    def encode(self, x):
        """Encode gene expression to latent space."""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode from latent space to gene expression."""
        return self.decoder(z)
    
    def predict_condition(self, z):
        """Predict the condition from latent representation."""
        return self.adversary(z)
    
    def forward(self, x, condition_idx=None, return_latent=False):
        """Forward pass through the model."""
        # Encode to latent space - this is the basal representation
        z_basal = self.encode(x)
        
        # Create modified latent representation based on condition
        if condition_idx is not None:
            # If condition is provided, add its embedding to latent
            condition_emb = self.condition_embeddings(condition_idx)
            z_modified = z_basal + condition_emb
        else:
            z_modified = z_basal
            
        # Decode to reconstruction
        recon = self.decode(z_modified)
        
        # Split into mean and variance
        dim = recon.size(1) // 2
        mean = recon[:, :dim]
        log_var = recon[:, dim:]
        var = torch.exp(log_var)
        
        if return_latent:
            return mean, var, z_basal
        return mean, var
    
    def predict(self, genes, condition_idx=None):
        """Predict gene expression under a specific condition."""
        self.eval()
        with torch.no_grad():
            mean, var = self.forward(genes, condition_idx)
            return mean, var
            
    def update_step(self, genes, condition_idx, optimize_adversary=True):
        """Perform a single update step."""
        # Get latent representation (basal state without condition)
        mean, var, z_basal = self.forward(genes, condition_idx=None, return_latent=True)
        
        # Reconstruction loss
        recon_loss = self.loss_reconstruction(mean, genes, var)
        
        # Adversary loss (predict condition from latent)
        adv_pred = self.predict_condition(z_basal)
        adv_loss = self.loss_adversary(adv_pred, condition_idx)
        
        # Compute gradient penalty (as in original ChemCPA)
        grad_penalty = torch.tensor(0.0, device=self.device)
        if not optimize_adversary:
            # Only compute gradient penalty when updating the main model
            def compute_gradient_penalty(out, x):
                """Compute gradient penalty for the adversary"""
                # Get gradient of output with respect to input
                grads = torch.autograd.grad(out.sum(), x, create_graph=True)[0]
                # Return the squared gradient norm
                return (grads ** 2).mean()
            
            # Add gradient penalty for latent representation
            grad_penalty = compute_gradient_penalty(adv_pred, z_basal)
            
        if optimize_adversary:
            loss = adv_loss
        else:
            loss = recon_loss - self.reg_adversary * adv_loss + self.penalty_adversary * grad_penalty
            
        return loss, {
            "recon_loss": recon_loss.item(), 
            "adv_loss": adv_loss.item(),
            "grad_penalty": grad_penalty.item()
        }
    
    def to(self, device):
        """Move model to specified device."""
        self.device = device
        return super(SimplifiedChemCPA, self).to(device)


# Lightning module for training
class SimplifiedChemCPAModule(L.LightningModule):
    def __init__(self, model, 
                 adversary_steps=3,
                 lr_encoder=0.0015751320499779737,
                 lr_decoder=0.0015751320499779737, 
                 lr_adversary=0.0011926173789223548,
                 wd_encoder=6.251373574521742e-7,
                 wd_decoder=6.251373574521742e-7,
                 wd_adversary=0.000009846738873614555,
                 step_size_lr=6,
                 patience=50):
        super(SimplifiedChemCPAModule, self).__init__()
        self.model = model
        self.adversary_steps = adversary_steps
        self.lr_encoder = lr_encoder
        self.lr_decoder = lr_decoder
        self.lr_adversary = lr_adversary
        self.wd_encoder = wd_encoder
        self.wd_decoder = wd_decoder
        self.wd_adversary = wd_adversary
        self.step_size_lr = step_size_lr
        self.patience = patience
        self.automatic_optimization = False
        
    def forward(self, x, condition_idx=None):
        return self.model(x, condition_idx)
    
    def training_step(self, batch, batch_idx):
        # Get optimizers
        optimizers = self.optimizers()
        opt_encoder = optimizers[0]
        opt_decoder = optimizers[1]
        opt_adversary = optimizers[2]
        
        # Unpack batch - gene expression and condition index
        genes, condition_idx = batch
        
        # Decide whether to update adversary or encoder/decoder
        optimize_adversary = (self.global_step % (self.adversary_steps + 1)) == 0
        
        if optimize_adversary:
            # Update adversary
            opt_adversary.zero_grad()
            loss, loss_dict = self.model.update_step(genes, condition_idx, optimize_adversary=True)
            self.manual_backward(loss)
            opt_adversary.step()
        else:
            # Update encoder and decoder
            opt_encoder.zero_grad()
            opt_decoder.zero_grad()
            loss, loss_dict = self.model.update_step(genes, condition_idx, optimize_adversary=False)
            self.manual_backward(loss)
            
            # Gradient clipping as in the original implementation
            self.clip_gradients(opt_encoder, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            self.clip_gradients(opt_decoder, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            
            opt_encoder.step()
            opt_decoder.step()
        
        # Log the loss
        self.log("loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        for k, v in loss_dict.items():
            self.log(k, v, on_epoch=True, on_step=False)
            
        return loss
    
    def configure_optimizers(self):
        # Separate optimizers for encoder/decoder and adversary with weight decay
        encoder_params = list(self.model.encoder.parameters())
        decoder_params = list(self.model.decoder.parameters()) + list(self.model.condition_embeddings.parameters())
        adversary_params = list(self.model.adversary.parameters())
        
        opt_encoder = torch.optim.Adam(encoder_params, lr=self.lr_encoder, weight_decay=self.wd_encoder)
        opt_decoder = torch.optim.Adam(decoder_params, lr=self.lr_decoder, weight_decay=self.wd_decoder)
        opt_adversary = torch.optim.Adam(adversary_params, lr=self.lr_adversary, weight_decay=self.wd_adversary)
        
        return [opt_encoder, opt_decoder, opt_adversary]


# Constants
GENE_PERTURBATION_PATH = "/fs/cbcb-scratch/cys/w1ot/gene_perturbation_experiments"
OUTPUT_DIR = "/fs/cbcb-scratch/cys/w1ot/test_baselines/gene_perturbation_experiments/chemcpa"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 200  # Reduced for faster training


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


# DataModule for our simplified ChemCPA
class SimpleDataModule(L.LightningDataModule):
    def __init__(self, source_train, target_train, source_test, target_test, 
                perturbation_attribute, batch_size=128):
        super().__init__()
        self.source_train = source_train
        self.target_train = target_train
        self.source_test = source_test
        self.target_test = target_test
        self.perturbation_attribute = perturbation_attribute
        self.batch_size = batch_size
        
    def setup(self, stage=None):
        # Combine source and target train data
        self.combined_train = ad.concat([self.source_train, self.target_train])
        
        # Create condition mapping (0=control, 1=perturbation)
        self.condition_map = {"control": 0, 
                             self.target_train.obs[self.perturbation_attribute].unique()[0]: 1}
        
        # Extract gene expression and condition indices
        self.train_genes = torch.tensor(ensure_numpy(self.combined_train.X), dtype=torch.float32)
        self.train_conditions = torch.tensor(
            [self.condition_map[c] for c in self.combined_train.obs[self.perturbation_attribute]], 
            dtype=torch.long
        )
        
        # Also prepare test data
        self.source_test_genes = torch.tensor(ensure_numpy(self.source_test.X), dtype=torch.float32)
        self.target_test_genes = torch.tensor(ensure_numpy(self.target_test.X), dtype=torch.float32)
        
    def train_dataloader(self):
        # Create dataset
        dataset = torch.utils.data.TensorDataset(self.train_genes, self.train_conditions)
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0
        )


def run_chemcpa_on_perturbation(dataset_name, pert_dir, pert_path):
    """Run simplified ChemCPA on a specific gene perturbation dataset."""
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

    if np.min(source_train.X) < 0 or np.min(target_train.X) < 0:
        POSITIVE_ONLY = False
    else:
        POSITIVE_ONLY = True
    
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
    
    # Create data module
    data_module = SimpleDataModule(
        source_train=source_train,
        target_train=target_train,
        source_test=source_test,
        target_test=target_test,
        perturbation_attribute=perturbation_attribute,
        batch_size=256
    )
    data_module.setup()
    
    # Set latent dimension based on dataset
    if "4i" in dataset_name:
        LATENT_DIM = 8
    else:
        LATENT_DIM = 50
    
    print(f"Using latent dimension: {LATENT_DIM} for dataset: {dataset_name}")
    
    
    model = SimplifiedChemCPA(
        num_genes=source_train.shape[1],
        num_conditions=2,  # Just control and perturbation
        latent_dim=LATENT_DIM,
        encoder_width=256,
        encoder_depth=4,
        decoder_width=256,
        decoder_depth=4,
        adversary_width=128,
        adversary_depth=3,
        reg_adversary=0,
        penalty_adversary=100,
        dropout=0.262378,
        device=DEVICE
    )
    
    # Create the Lightning module
    lightning_module = SimplifiedChemCPAModule(
        model=model,
        adversary_steps=6,
        lr_encoder=1e-3,
        lr_decoder=1e-3,
        lr_adversary=1e-3,
        wd_encoder=1e-7,
        wd_decoder=1e-7,
        wd_adversary=1e-7,
        step_size_lr=6,
        patience=50
    )
    
    # Set up callbacks for training
    checkpoint_callback = ModelCheckpoint(
        dirpath=dataset_output_dir,
        filename='best-model',
        save_top_k=1,
        monitor='loss',
        mode='min'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='loss',
        patience=50,
        mode='min'
    )
    
    # Set up the trainer
    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=NUM_EPOCHS,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_progress_bar=True,
        logger=False,
        enable_model_summary=True,
        log_every_n_steps=10,
    )
    
    # Train the model
    print(f"Training simplified ChemCPA for up to {NUM_EPOCHS} epochs...")
    trainer.fit(lightning_module, data_module)
    print("Training completed.")
    
    # Load the best model for evaluation
    best_model_path = checkpoint_callback.best_model_path
    lightning_module = SimplifiedChemCPAModule.load_from_checkpoint(
        best_model_path,
        model=model  # Pass the model architecture
    )
    lightning_module.eval()
    
    # Move test data to the proper device
    source_test_genes = data_module.source_test_genes.to(lightning_module.device)
    
    # Generate predictions for control cells treated with the perturbation
    print("Generating counterfactual predictions...")
    with torch.no_grad():
        # Condition index 1 = perturbation
        perturbation_condition = torch.ones(len(source_test_genes), dtype=torch.long, device=lightning_module.device)
        mean_predictions, _ = lightning_module.model.predict(source_test_genes, perturbation_condition)
        
        # Convert to numpy array
        perturbed_predictions_np = mean_predictions.cpu().numpy()
    
    print(f"Generated predictions shape: {perturbed_predictions_np.shape}")
    if POSITIVE_ONLY:
        # Ensure all predictions are non-negative
        perturbed_predictions_np = np.clip(perturbed_predictions_np, 0, np.inf)
    
    # Create AnnData object for evaluation
    transported_adata = ad.AnnData(X=perturbed_predictions_np, var=source_test.var)
    
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
    transported_adata.write(os.path.join(predictions_dir, 'chemcpa_predicted.h5ad'))
    print(f"Saved prediction to {os.path.join(predictions_dir, 'chemcpa_predicted.h5ad')}")
    
    # Create result dictionary
    results = {
        "model": "simplified_chemcpa",
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
    """Main function to run ChemCPA on gene perturbation datasets."""
    # Force flushing of print statements to ensure log contains output
    import sys
    import time
    
    print("Starting simplified ChemCPA evaluation on gene perturbation data...", flush=True)
    print(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Python version: {sys.version}", flush=True)
    print(f"Working directory: {os.getcwd()}", flush=True)
    print(f"Using device: {DEVICE}", flush=True)
    sys.stdout.flush()
    
    setup()
    
    # Get all gene perturbation datasets
    perturbation_paths = get_gene_perturbation_datasets()
    
    if not perturbation_paths:
        print("No gene perturbation datasets found in the directory.")
        return
    
    print(f"Found {len(perturbation_paths)} gene perturbation datasets.")
    
    # Run simplified ChemCPA on each perturbation
    results = []
    for (dataset_name, pert_dir), pert_path in perturbation_paths.items():
        pert_result = run_chemcpa_on_perturbation(dataset_name, pert_dir, pert_path)
        results.append(pert_result)
    
    # Summarize results
    print("\nEvaluation completed!")
    print(f"Processed {len(results)} out of {len(perturbation_paths)} perturbations successfully.")


if __name__ == "__main__":
    main() 