import os
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from anndata import AnnData
from typing import Any
from .utils import ensure_numpy, normalize_and_log_transform

class VAEModel(nn.Module):
    def __init__(self, encoder: nn.Module, fc_mu: nn.Module, fc_logvar: nn.Module, decoder: nn.Module):
        super(VAEModel, self).__init__()
        self.encoder = encoder
        self.fc_mu = fc_mu
        self.fc_logvar = fc_logvar
        self.decoder = decoder
                            
    def vae_encode(self, x: torch.Tensor) -> tuple:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def ae_encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_mu(self.encoder(x))
                            
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
                            
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
                            
    def forward(self, x: torch.Tensor) -> tuple:
        mu, logvar = self.vae_encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
                            
    def loss_function(self, recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, alpha: float) -> tuple:
        MSE = nn.functional.mse_loss(recon, x, reduction='mean')
        KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        ELBO = MSE + alpha * KLD
        return ELBO, MSE, KLD

class VAE:
    def __init__(self, alpha: float, device: str, output_dir: str):
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.best_val_elbo = float('inf')
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        self.train_data = None
        self.val_data = None
        self.alpha = alpha
        self.p = None
        self.checkpoint = None

    def setup_anndata(self, anndata: AnnData, layer: str = None) -> None:
        """
        Set up the AnnData object and specify the data layer to use.

        Parameters:
            anndata (AnnData): The AnnData object containing the dataset.
            layer (str): The layer of the AnnData object to use for training.
        """
        if layer is None:
            self.data = anndata.X
        else:
            self.data = anndata.layers[layer]

        self.data = ensure_numpy(self.data)
        
        # Check if the data is not log transformed
        if self.data.max() > 50:
            # Data is not log-transformed, perform normalization and log transformation
            self.data = normalize_and_log_transform(self.data)

        self.data = torch.FloatTensor(self.data).to(self.device)
        self.p = self.data.shape[1]

    def setup_model(self, hidden_layers: list, latent_dim: int) -> None:
        """
        Initialize the VAE model architecture.

        Parameters:
            hidden_layers (list): A list of integers specifying the number of neurons in each hidden layer.
            latent_dim (int): The dimensionality of the latent space.
        """
        layers = []
        input_dim = self.data.shape[1]
        
        # Encoder
        for h in hidden_layers:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        self.encoder = nn.Sequential(*layers)
        
        # Latent space
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        input_dim = latent_dim
        for h in reversed(hidden_layers):
            decoder_layers.append(nn.Linear(input_dim, h))
            decoder_layers.append(nn.ReLU())
            input_dim = h
        decoder_layers.append(nn.Linear(input_dim, self.data.shape[1]))
        decoder_layers.append(nn.ReLU())
        
        self.decoder = nn.Sequential(*decoder_layers)
                
        self.model = VAEModel(self.encoder, self.fc_mu, self.fc_logvar, self.decoder).to(self.device)

    def create_dataset(self, validation_split: float = 0.2) -> None:
        """
        Create training and validation datasets and initialize permutation indices.

        Parameters:
            validation_split (float): The proportion of data to use for validation.
        """
        dataset_size = self.data.size(0)
        if validation_split > 0:
            val_size = int(validation_split * dataset_size)
            train_size = dataset_size - val_size
            self.train_data, self.val_data = torch.split(self.data, [train_size, val_size])
        else:
            self.train_data = self.data
            self.val_data = None

        # Initialize permutation indices and counters for training and validation data
        self.train_indices = torch.randperm(self.train_data.size(0)).to(self.device)
        self.train_idx = 0

        if self.val_data is not None:
            self.val_indices = torch.randperm(self.val_data.size(0)).to(self.device)
            self.val_idx = 0

    def sample_train_batch(self, batch_size: int) -> torch.Tensor:
        if batch_size > self.train_data.size(0):
            batch_size = self.train_data.size(0)
        
        if self.train_idx + batch_size > self.train_data.size(0):
            # Reshuffle and reset index
            self.train_indices = torch.randperm(self.train_data.size(0)).to(self.device)
            self.train_idx = 0

        idx = self.train_indices[self.train_idx:self.train_idx + batch_size]
        self.train_idx += batch_size
        return self.train_data[idx]

    def sample_val_batch(self, batch_size: int) -> torch.Tensor:
        if self.val_data is None:
            return None
        if batch_size > self.val_data.size(0):
            batch_size = self.val_data.size(0)
        
        if self.val_idx + batch_size > self.val_data.size(0):
            # Reshuffle and reset index
            self.val_indices = torch.randperm(self.val_data.size(0)).to(self.device)
            self.val_idx = 0

        idx = self.val_indices[self.val_idx:self.val_idx + batch_size]
        self.val_idx += batch_size
        return self.val_data[idx]

    def train(self, 
              num_iters: int = 250000, 
              batch_size: int = 256, 
              lr: float = 1e-3,
              validation_split: float = 0.2,
              resume_from_checkpoint: bool = False,
              checkpoint_interval: int = 2500,
              lr_step_size: int = 100000,
              lr_gamma: float = 0.5) -> None:
        """
        Train the VAE model.

        Parameters:
            num_iters (int): The number of iterations to train the model.
            batch_size (int): The batch size for training.
            lr (float): The learning rate for the optimizer.
            validation_split (float): The proportion of data to use for validation.
            resume_from_checkpoint (bool): Whether to resume training from a checkpoint.
            checkpoint_interval (int): The number of iterations between checkpoints.
            lr_step_size (int): Number of iterations before learning rate adjustment.
            lr_gamma (float): Multiplicative factor of learning rate decay.
        """
        checkpoint_path = os.path.join(self.output_dir, 'vae_checkpoint.pth')

        # Create datasets and initialize permutation indices
        self.create_dataset(validation_split)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
        
        start_iter = 0
        self.best_val_elbo = float('inf')
        if resume_from_checkpoint and os.path.exists(checkpoint_path):
            try:
                self.load(checkpoint_path)
                optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(self.checkpoint['scheduler_state_dict'])
                start_iter = self.checkpoint['iteration']
                self.best_val_elbo = self.checkpoint['best_val_elbo']
                print(f"Resuming from iteration {start_iter} with best validation ELBO {self.best_val_elbo:.4f}")
            except:
                print("Failed to resume from checkpoint. Starting from scratch.")

        progress_bar = tqdm(range(start_iter, num_iters), desc="Training")
        for iteration in progress_bar:
            self.model.train()
            # Training step
            train_batch = self.sample_train_batch(batch_size)
            optimizer.zero_grad()

            if self.alpha > 0:
                recon, mu, logvar = self.model(train_batch)
                loss, recon_loss, kld_loss = self.model.loss_function(recon, train_batch, mu, logvar, self.alpha)
            else:
                z = self.model.ae_encode(train_batch)
                recon = self.model.decode(z)
                loss = nn.functional.mse_loss(recon, train_batch)
                
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Validation step
            if self.val_data is not None:
                self.model.eval()
                with torch.no_grad():
                    val_batch = self.sample_val_batch(batch_size)

                    if self.alpha > 0:
                        val_recon, val_mu, val_logvar = self.model(val_batch)
                        val_loss, val_recon_loss, val_kld_loss = self.model.loss_function(val_recon, val_batch, val_mu, val_logvar, self.alpha)
                    else:
                        z = self.model.ae_encode(val_batch)
                        val_recon = self.model.decode(z)
                        val_loss = nn.functional.mse_loss(val_recon, val_batch)
                    
                if val_loss < self.best_val_elbo and (iteration+1) % checkpoint_interval == 0:
                    self.best_val_elbo = val_loss 
                    self._save_checkpoint(iteration + 1, optimizer, scheduler, checkpoint_path)
                    
                progress_bar.set_postfix({'loss': val_loss.item()})
            else:
                progress_bar.set_postfix({'loss': loss.item()})
                if (iteration+1) % checkpoint_interval == 0:
                    self._save_checkpoint(iteration + 1, optimizer, scheduler, checkpoint_path)
            
        self.load(checkpoint_path)
        print(f'Training completed. Best ELBO: {self.best_val_elbo:.4f}')

    def _save_checkpoint(self, iteration, optimizer, scheduler, checkpoint_path):
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_elbo': self.best_val_elbo
        }
        torch.save(checkpoint, checkpoint_path)

    def load(self, checkpoint_path: str) -> None:
        """
        Load a checkpoint into the model.

        Parameters:
            checkpoint_path (str): Path to the checkpoint file.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.best_val_elbo = self.checkpoint['best_val_elbo']
        
                            
    def get_latent_representation(self, data=None) -> np.ndarray:
        """
        Obtain the latent representations from the VAE.

        Returns:
            np.ndarray: The latent representations of the data.
        """
        try:
            checkpoint_path = os.path.join(self.output_dir, 'vae_checkpoint.pth')
            self.load(checkpoint_path)
        except:
            try:
                checkpoint_path = os.path.join(self.output_dir, '../scgen/vae_checkpoint.pth')
                self.load(checkpoint_path)
            except:
                raise ValueError("No checkpoint found. Please load a checkpoint first.")
        
        if data is None:
            self.model.eval()
            with torch.no_grad():
                if self.alpha > 0:
                    mu, logvar = self.model.vae_encode(data)
                    z = mu
                else:
                    z = self.model.ae_encode(data)
                return z.cpu().numpy()
        else:
            # data format check
            if isinstance(data, AnnData):
                data = ensure_numpy(data.X)
            elif isinstance(data, np.ndarray):
                pass
            elif isinstance(data, torch.Tensor):
                pass
            else:
                raise ValueError(f"Unknown data type: {type(data)}")
            # check if data is normalized and log transformed
            if np.max(data) > 50:
                data = normalize_and_log_transform(data)
            self.model.eval()
            with torch.no_grad():
                data = torch.tensor(data, dtype=torch.float32).to(self.device)

                if self.alpha > 0:
                    mu, logvar = self.model.vae_encode(data)
                    z = mu
                else:
                    z = self.model.ae_encode(data)
                return z.cpu().numpy()
                            
    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from latent representations.

        Parameters:
            z (np.ndarray): The latent vectors.

        Returns:
            np.ndarray: The reconstructed data.
        """
        try:
            checkpoint_path = os.path.join(self.output_dir, 'vae_checkpoint.pth')
            self.load(checkpoint_path)
        except:
            try:
                checkpoint_path = os.path.join(self.output_dir, '../scgen/vae_checkpoint.pth')
                self.load(checkpoint_path)
            except:
                raise ValueError("No checkpoint found. Please load a checkpoint first.")
        
        self.model.eval()
        with torch.no_grad():
            z_tensor = torch.tensor(z, dtype=torch.float32).to(self.device)
            recon = self.model.decode(z_tensor)
            return recon.cpu().numpy()