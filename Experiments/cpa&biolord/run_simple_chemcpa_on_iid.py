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

# Add parent directory to path to import chemCPA
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scipy.stats import pearsonr
from chemCPA.model import MLP  # Reuse the MLP implementation from chemCPA


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

    # Print first few values for debugging
    print(f"Transported mean (first 5): {transported_mean[:5]}")
    print(f"Target mean (first 5): {target_mean[:5]}")
        
    # Pearson correlation coefficient between means
    corr_coeff, _ = pearsonr(transported_mean, target_mean)
    r2 = corr_coeff ** 2
        
    # L2 distance between means
    l2_dist = np.linalg.norm(transported_mean - target_mean)
        
    # MMD distance
    gammas = np.logspace(1, -3, num=50)
    mmd_dist = np.mean([mmd_distance(target_data, transported_data, g) for g in gammas])
        
    return r2, l2_dist, mmd_dist


# Simplified ChemCPA model for IID experiments
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
IID_EXPERIMENTS_PATH = "/fs/cbcb-scratch/cys/w1ot/iid_experiments"
OUTPUT_DIR = "/fs/cbcb-scratch/cys/w1ot/test_baselines/iid_experiments/chemcpa"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 200  # Reduced for faster training


def setup():
    """Create output directory and prepare for testing."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Setup complete. Output will be saved to {OUTPUT_DIR}", flush=True)
    print(f"Using device: {DEVICE}", flush=True)


def get_perturbation_datasets():
    """Find datasets in the iid_experiments directory, limited to specific datasets."""
    # Only include t"hese specific datasets
    dataset_dirs = ["4i-melanoma-8h-48"] # ,
    
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


def run_parameter_sweep():
    """
    Run a parameter sweep to find the best hyperparameters for SimpleChemCPA based on MMD score.
    Randomly selects one perturbation task to perform the search on.
    """
    print("Starting parameter sweep for SimpleChemCPA...", flush=True)
    
    # Set up environment
    setup()
    
    # Get all perturbation datasets
    perturbation_paths = get_perturbation_datasets()
    
    if not perturbation_paths:
        print("No perturbation datasets found in the iid_experiments directory.")
        return None
    
    # Randomly select one perturbation task
    random.seed(42)  # For reproducibility
    selected_key = random.choice(list(perturbation_paths.keys()))
    dataset_name, pert_dir = selected_key
    pert_path = perturbation_paths[selected_key]
    
    print(f"Randomly selected perturbation task: {dataset_name} - {pert_dir}")
    
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

    num_cells = min(source_train.shape[0], target_train.shape[0])
    source_train = source_train[:num_cells]
    target_train = target_train[:num_cells]
    print("Balanced number of cells in source and target train sets", source_train.shape, target_train.shape)

    
    print(f"Data loaded:")
    print(f"  Source train: {source_train.shape}")
    print(f"  Source test: {source_test.shape}")
    print(f"  Target train: {target_train.shape}")
    print(f"  Target test: {target_test.shape}")

    
    # Ensure perturbation attribute is properly set
    if perturbation_attribute not in source_train.obs.columns:
        print(f"Adding {perturbation_attribute} column to AnnData objects")
        source_train.obs[perturbation_attribute] = 'control'
        source_test.obs[perturbation_attribute] = 'control'
        target_train.obs[perturbation_attribute] = perturbation_name
        target_test.obs[perturbation_attribute] = perturbation_name
    
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
    elif "sciplex" in dataset_name:
        LATENT_DIM = 50
    else:
        LATENT_DIM = 32  # Default value
    
    # Define parameter grid for sweep
    # Focus on key parameters that affect model performance
    reduced_param_grid  = {
        # Model parameters
        "encoder_width": [256],
        "encoder_depth": [4],
        "decoder_width": [256],
        "decoder_depth": [4],
        "adversary_width": [128],
        "adversary_depth": [3],
        "reg_adversary": [0],
        "penalty_adversary": [5, 10, 20, 50, 100],
        "dropout": [0.1],
        "adversary_steps": [1, 3, 6],
        "lr_encoder": [1e-4],
        "lr_decoder": [1e-4],
        "lr_adversary": [1e-4],
        "wd_encoder": [1e-7],
        "wd_decoder": [1e-7], 
        "wd_adversary": [1e-7],
    }
    
    # Prepare for parameter sweep
    all_results = []
    
    # Generate parameter combinations
    param_combinations = list(product(*reduced_param_grid.values()))
    total_combinations = len(param_combinations)
    print(f"Testing {total_combinations} parameter combinations")
    
    # Use a reduced number of epochs for the sweep
    sweep_epochs = 100
    
    # Prepare to evaluate using top differential genes
    top_k = 50
    if top_k > target_test.shape[1]:
        gene_list = None
        print("Using all genes for evaluation")
    else:
        try:
            # Try to load existing marker genes
            try:
                perturbation = target_test.obs[perturbation_attribute].unique()[0]
                gene_list = target_test.varm['marker_genes-drug-rank'][perturbation].sort_values()[:top_k].index
                print("Loaded existing gene list")
            except:
                # If loading fails, compute differential expression
                print("Computing new differential gene list...")
                combined_test = ad.concat([source_test, target_test])
                sc.tl.rank_genes_groups(combined_test, perturbation_attribute, reference='control')
                gene_list = combined_test.uns['rank_genes_groups']['names'][perturbation_name][:top_k]
                print(f"Created new gene list with top {top_k} differential genes")
        except Exception as e:
            print(f"Warning: Could not compute differential genes: {e}")
            gene_list = None
    
    # Run sweep
    for i, params in enumerate(param_combinations):
        # Create parameter dictionary from current combination
        param_dict = {k: v for k, v in zip(reduced_param_grid.keys(), params)}
        
        print(f"\nTesting parameter set {i+1}/{total_combinations}:")
        for k, v in param_dict.items():
            print(f"  {k}: {v}")
        
        # Create the model with current parameters
        model = SimplifiedChemCPA(
            num_genes=source_train.shape[1],
            num_conditions=2,  # Just control and perturbation
            latent_dim=LATENT_DIM,
            encoder_width=param_dict["encoder_width"],
            encoder_depth=param_dict["encoder_depth"],
            decoder_width=param_dict["decoder_width"],
            decoder_depth=param_dict["decoder_depth"],
            adversary_width=param_dict["adversary_width"],
            adversary_depth=param_dict["adversary_depth"],
            reg_adversary=param_dict["reg_adversary"],
            penalty_adversary=param_dict["penalty_adversary"],
            dropout=param_dict["dropout"],
            device=DEVICE
        )
        
        # Create the Lightning module with current parameters
        lightning_module = SimplifiedChemCPAModule(
            model=model,
            adversary_steps=param_dict["adversary_steps"],
            lr_encoder=param_dict["lr_encoder"],
            lr_decoder=param_dict["lr_decoder"],
            lr_adversary=param_dict["lr_adversary"],
            wd_encoder=param_dict["wd_encoder"],
            wd_decoder=param_dict["wd_decoder"],
            wd_adversary=param_dict["wd_adversary"],
            step_size_lr=6,  # Fixed for sweep
            patience=20  # Lower patience for faster sweep
        )
        
        # Set up callbacks for training
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(sweep_output_dir, f"run_{i+1}"),
            filename='best-model',
            save_top_k=1,
            monitor='loss',
            mode='min'
        )
        
        early_stop_callback = EarlyStopping(
            monitor='loss',
            patience=20,  # Lower patience for faster sweep
            mode='min'
        )
        
        # Set up the trainer with reduced epochs
        trainer = L.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_epochs=sweep_epochs,
            callbacks=[checkpoint_callback, early_stop_callback],
            enable_progress_bar=False,
            logger=False,
            enable_model_summary=False,
            log_every_n_steps=10,
        )
        
        try:
            # Train the model
            print(f"Training model {i+1}/{total_combinations} for up to {sweep_epochs} epochs...")
            trainer.fit(lightning_module, data_module)
            print("Training completed.")
            
            # Load the best model for evaluation
            try:
                best_model_path = checkpoint_callback.best_model_path
                if best_model_path and os.path.exists(best_model_path):
                    lightning_module = SimplifiedChemCPAModule.load_from_checkpoint(
                        best_model_path,
                        model=model  # Pass the model architecture
                    )
                    lightning_module.eval()
                else:
                    print("No best model checkpoint found, using current model")
                    lightning_module.eval()
            except Exception as e:
                print(f"Error loading best model, using current model: {e}")
                lightning_module.eval()
            
            # Move test data to the proper device
            source_test_genes = data_module.source_test_genes.to(lightning_module.device)
            
            # Generate predictions
            print("Generating predictions...")
            with torch.no_grad():
                # Condition index 1 = perturbation
                perturbation_condition = torch.ones(len(source_test_genes), dtype=torch.long, device=lightning_module.device)
                mean_predictions, _ = lightning_module.model.predict(source_test_genes, perturbation_condition)
                perturbed_predictions_np = mean_predictions.cpu().numpy()
            
            # Create AnnData object for evaluation
            transported_adata = ad.AnnData(X=perturbed_predictions_np, var=source_test.var)
            
            # Calculate evaluation metrics
            r2_score, l2norm, mmd_score = metrics(
                transported=transported_adata,
                target=target_test,
                gene_list=gene_list
            )
            
            # Store results
            param_result = {
                "model": "simplified_chemcpa",
                "param_set": i+1,
                "perturbation": perturbation_name,
                "cell_r2": r2_score,
                "cell_l2": l2norm,
                "cell_mmd": mmd_score,
            }
            # Add parameters to result
            for k, v in param_dict.items():
                param_result[k] = v
            
            all_results.append(param_result)
            
            print(f"Parameter set {i+1} results:")
            print(f"  R2 score: {r2_score:.6f}")
            print(f"  L2 norm: {l2norm:.6f}")
            print(f"  MMD: {mmd_score:.6f}")
            
            # Save intermediate results
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(os.path.join(sweep_output_dir, "param_sweep_results.csv"), index=False)
            
        except Exception as e:
            print(f"Error during training/evaluation for parameter set {i+1}: {e}")
            # Add this failed run to results with NaN values
            param_result = {
                "model": "simplified_chemcpa",
                "param_set": i+1,
                "perturbation": perturbation_name,
                "cell_r2": float('nan'),
                "cell_l2": float('nan'),
                "cell_mmd": float('nan'),
                "failed": True,
                "error": str(e)
            }
            for k, v in param_dict.items():
                param_result[k] = v
            
            all_results.append(param_result)
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(os.path.join(sweep_output_dir, "param_sweep_results.csv"), index=False)
    
    # Find best parameters based on MMD score if we have results
    if all_results:
        results_df = pd.DataFrame(all_results)
        # Filter out failed runs
        valid_results = results_df[~results_df['cell_mmd'].isna()]
        
        if not valid_results.empty:
            best_params_idx = valid_results['cell_mmd'].idxmin()
            best_params = valid_results.iloc[best_params_idx].to_dict()
            
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
        else:
            print("No valid results found in parameter sweep.")
            return None
    else:
        print("No results obtained from parameter sweep.")
        return None


def run_chemcpa_on_perturbation(dataset_name, pert_dir, pert_path, custom_params=None):
    """Run simplified ChemCPA on a specific perturbation dataset."""
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

    print(np.max(source_train.X), np.min(source_train.X))
    print(np.max(target_train.X), np.min(target_train.X))
    
    if np.min(source_train.X) < 0 or np.min(target_train.X) < 0:
        POSITIVE_ONLY = False
    else:
        POSITIVE_ONLY = True

    print(f"Data loaded:")
    print(f"  Source train: {source_train.shape}")
    print(f"  Source test: {source_test.shape}")
    print(f"  Target train: {target_train.shape}")
    print(f"  Target test: {target_test.shape}")


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
    
    
    # Ensure perturbation attribute is properly set
    # Set source cells to 'control' and target cells to the perturbation name
    if perturbation_attribute not in source_train.obs.columns:
        print(f"Adding {perturbation_attribute} column to AnnData objects")
        source_train.obs[perturbation_attribute] = 'control'
        source_test.obs[perturbation_attribute] = 'control'
        target_train.obs[perturbation_attribute] = perturbation_name
        target_test.obs[perturbation_attribute] = perturbation_name
    
    # Set latent dimension based on dataset
    if "4i" in dataset_name:
        LATENT_DIM = 8
    elif "sciplex" in dataset_name:
        LATENT_DIM = 50
    else:
        LATENT_DIM = 32  # Default value
    
    print(f"Using latent dimension: {LATENT_DIM} for dataset: {dataset_name}")
    
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
    
    # Use custom parameters if provided, otherwise use defaults
    if custom_params is None:
        # Create the simplified model with default hyperparameters
        model = SimplifiedChemCPA(
            num_genes=source_train.shape[1],
            num_conditions=2,  # Just control and perturbation
            latent_dim=LATENT_DIM,
            encoder_width=256,
            encoder_depth=4,    # default from config
            decoder_width=256,  # default from config
            decoder_depth=4,    # default from config
            adversary_width=128,  # default from config
            adversary_depth=3,   # default from config
            reg_adversary=9.100951626404369,
            penalty_adversary=0.4550475813202185,  # default from config
            dropout=0.262378,    # default from config
            device=DEVICE
        )
        
        # Create the Lightning module with default hyperparameters
        lightning_module = SimplifiedChemCPAModule(
            model=model,
            adversary_steps=3,  # default from config
            lr_encoder=1e-3,  # default from config
            lr_decoder=1e-3,  # default from config
            lr_adversary=1e-3,  # default from config
            wd_encoder=1e-7,  # default from config
            wd_decoder=1e-7,  # default from config
            wd_adversary=1e-7,  # default from config
            step_size_lr=6,  # default from config
            patience=50   # default from config
        )
    else:
        print("Using custom parameters from parameter sweep")
        # Create the simplified model with custom hyperparameters
        model = SimplifiedChemCPA(
            num_genes=source_train.shape[1],
            num_conditions=2,  # Just control and perturbation
            latent_dim=LATENT_DIM,
            encoder_width=custom_params.get("encoder_width", 256),
            encoder_depth=custom_params.get("encoder_depth", 4),
            decoder_width=custom_params.get("decoder_width", 256),
            decoder_depth=custom_params.get("decoder_depth", 4),
            adversary_width=custom_params.get("adversary_width", 128),
            adversary_depth=custom_params.get("adversary_depth", 3),
            reg_adversary=custom_params.get("reg_adversary", 9.100951626404369),
            penalty_adversary=custom_params.get("penalty_adversary", 0.4550475813202185),
            dropout=custom_params.get("dropout", 0.262378),
            device=DEVICE
        )
        
        # Create the Lightning module with custom hyperparameters
        lightning_module = SimplifiedChemCPAModule(
            model=model,
            adversary_steps=custom_params.get("adversary_steps", 3),
            lr_encoder=custom_params.get("lr_encoder", 1e-3),
            lr_decoder=custom_params.get("lr_decoder", 1e-3),
            lr_adversary=custom_params.get("lr_adversary", 1e-3),
            wd_encoder=custom_params.get("wd_encoder", 1e-6),
            wd_decoder=custom_params.get("wd_decoder", 1e-6),
            wd_adversary=custom_params.get("wd_adversary", 1e-6),
            step_size_lr=6,  # Not included in sweep
            patience=50   # Not included in sweep
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
        patience=50,  # default from config
        mode='min'
    )
    
    # Set up the trainer
    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=NUM_EPOCHS,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_progress_bar=False,  # Disable progress bar
        logger=False,  # No logging for simplicity
        enable_model_summary=False,  # Disable model summary
        log_every_n_steps=10,  # Log less frequently
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
        perturbed_predictions_np = np.clip(perturbed_predictions_np, 0, np.inf)
    
    # Prepare data for evaluation - use only top 50 most differential genes
    # Select top 50 features for evaluation
    top_k = 50
    if (top_k > target_test.shape[1]):
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
    transported_adata = ad.AnnData(X=perturbed_predictions_np, var=source_test.var)
    
    # Calculate evaluation metrics
    if gene_list is not None:
        print(f"Using top {len(gene_list)} differential genes for evaluation")
        r2_score, l2norm, mmd_score = metrics(
            transported=transported_adata,
            target=target_test,
            gene_list=gene_list
        )
    else:
        print("Using all genes for evaluation")
        r2_score, l2norm, mmd_score = metrics(
            transported=transported_adata,
            target=target_test
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
    """Main function to run ChemCPA on all perturbations."""
    # Force flushing of print statements to ensure log contains output
    import sys
    import time
    
    print("Starting simplified ChemCPA evaluation...", flush=True)
    print(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Python version: {sys.version}", flush=True)
    print(f"Working directory: {os.getcwd()}", flush=True)
    print(f"Using device: {DEVICE}", flush=True)
    sys.stdout.flush()
    
    setup()
    
    # First run parameter sweep to find the best hyperparameters
    print("Running parameter sweep to find optimal hyperparameters...")
    best_params = run_parameter_sweep()
    
    # Get all perturbation datasets
    perturbation_paths = get_perturbation_datasets()
    
    if not perturbation_paths:
        print("No perturbation datasets found in the iid_experiments directory.")
        return
    
    print(f"Found {len(perturbation_paths)} perturbation datasets.")
    
    # Run simplified ChemCPA on each perturbation using the best parameters
    results = []
    for (dataset_name, pert_dir), pert_path in perturbation_paths.items():
        pert_result = run_chemcpa_on_perturbation(dataset_name, pert_dir, pert_path, best_params)
        results.append(pert_result)
    
    # Summarize results
    print("\nEvaluation completed!")
    print(f"Processed {len(results)} out of {len(perturbation_paths)} perturbations successfully.")


if __name__ == "__main__":
    main() 