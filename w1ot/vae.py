import os
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from anndata import AnnData
from typing import Any
from .utils import ensure_numpy

class VAEModel(nn.Module):
    def __init__(self, encoder: nn.Module, fc_mu: nn.Module, fc_logvar: nn.Module, decoder: nn.Module):
        super(VAEModel, self).__init__()
        self.encoder = encoder
        self.fc_mu = fc_mu
        self.fc_logvar = fc_logvar
        self.decoder = decoder
                            
    def encode(self, x: torch.Tensor) -> tuple:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
                            
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
                            
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
                            
    def forward(self, x: torch.Tensor) -> tuple:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
                            
    def loss_function(self, recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, alpha: float) -> tuple:
        MSE = nn.functional.mse_loss(recon, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        ELBO = MSE + alpha * KLD
        return ELBO, MSE, KLD

class VAE:
    def __init__(self, device: str, output_dir: str):
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.best_val_elbo = float('inf')
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
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

        if isinstance(self.data, np.ndarray):
            pass
        elif hasattr(self.data, 'toarray'):
            self.data = self.data.toarray()

        self.data = self.normalize_and_log_transform(self.data)

        self.data = torch.FloatTensor(self.data).to(self.device)
    
    def normalize_and_log_transform(self, data: Any) -> None:
        """
        Normalize and log-transform the count data.
        """
        # normalize the data to make sure the sum of each row is 10000
        row_sums = np.sum(data, axis=1, keepdims=True) + 1e-5
        data = data / row_sums * 10000
        # log-transform the normalized data
        data = np.log1p(data)
        return data
                            
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
    
    def _sample_data(self, data: torch.Tensor, batch_size: int) -> torch.Tensor:
        idx = torch.randperm(data.size(0))[:batch_size]
        return data[idx]

    def train(self, 
              num_iters: int = 10000, 
              batch_size: int = 128, 
              lr: float = 1e-4,
              alpha: float = 1.0,
              validation_split: float = 0.2,
              resume_from_checkpoint: bool = False,
              checkpoint_interval: int = 1000) -> None:
        """
        Train the VAE model.

        Parameters:
            num_iters (int): The number of iterations to train the model.
            batch_size (int): The batch size for training.
            lr (float): The learning rate for the optimizer.
            alpha (float): The weight for the KLD loss.
            validation_split (float): The proportion of data to use for validation.
            resume_from_checkpoint (bool): Whether to resume training from a checkpoint.
        """
        checkpoint_path = os.path.join(self.output_dir, 'vae_checkpoint.pth')

        # Split the data into training and validation sets
        dataset_size = self.data.shape[0]
        if validation_split > 0:
            val_size = int(validation_split * dataset_size)
            train_size = dataset_size - val_size
            self.train_data, self.val_data = torch.split(self.data, [train_size, val_size])
        else:
            self.train_data = self.data
            self.val_data = None
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        start_iter = 0
        self.best_val_elbo = float('inf')
        if resume_from_checkpoint and os.path.exists(checkpoint_path):
            self.load(checkpoint_path)
            optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            start_iter = self.checkpoint['iteration']
            self.best_val_elbo = self.checkpoint['best_val_elbo']
            print(f"Resuming from iteration {start_iter} with best validation ELBO {self.best_val_elbo:.4f}")
        
        progress_bar = tqdm(range(start_iter, num_iters), desc="Training")
        for iteration in progress_bar:
            self.model.train()
            
            # Training step
            train_batch = self._sample_data(self.train_data, batch_size)
            optimizer.zero_grad()
            recon, mu, logvar = self.model(train_batch)
            loss, recon_loss, kld_loss = self.model.loss_function(recon, train_batch, mu, logvar, alpha)
            loss.backward()
            optimizer.step()
            
            # Validation step
            if self.val_data is not None:
                self.model.eval()
                with torch.no_grad():
                    val_batch = self._sample_data(self.val_data, batch_size)
                    val_recon, val_mu, val_logvar = self.model(val_batch)
                    val_loss, val_recon, val_kld = self.model.loss_function(val_recon, val_batch, val_mu, val_logvar, alpha)
            
            
                progress_bar.set_postfix({'ELBO': val_loss.item()})
            
            else:
                progress_bar.set_postfix({'ELBO': loss.item()})
            
            # save checkpoint periodically
            if (iteration + 1) % checkpoint_interval == 0:
                if self.val_data is not None and val_loss < self.best_val_elbo:
                    self.best_val_elbo = val_loss
                else:
                    self.best_val_elbo = loss
                self._save_checkpoint(iteration + 1, optimizer, checkpoint_path)
        
        print(f'Training completed. Best ELBO: {self.best_val_elbo:.4f}')
        # Load the best checkpoint after training
        self.load(checkpoint_path)
                            
    def _save_checkpoint(self, iteration, optimizer, checkpoint_path):
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
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
        print(f"Loaded checkpoint from iteration {self.checkpoint['iteration']} with best validation ELBO {self.best_val_elbo:.4f}")
                            
    def get_latent_representation(self, data=None) -> np.ndarray:
        """
        Obtain the latent representations from the VAE.

        Returns:
            np.ndarray: The latent representations of the data.
        """
        
        
        
        if data is None:
            self.model.eval()
            with torch.no_grad():
                mu, _ = self.model.encode(self.data)
                return mu.cpu().numpy()
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
            if np.max(data) > 100:
                data = self.normalize_and_log_transform(data)
            self.model.eval()
            with torch.no_grad():
                data = torch.tensor(data, dtype=torch.float32).to(self.device)
                mu, _ = self.model.encode(data)
                return mu.cpu().numpy()
                            
    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from latent representations.

        Parameters:
            z (np.ndarray): The latent vectors.

        Returns:
            np.ndarray: The reconstructed data.
        """
        self.model.eval()
        with torch.no_grad():
            z_tensor = torch.tensor(z, dtype=torch.float32).to(self.device)
            recon = self.model.decode(z_tensor)
            return recon.cpu().numpy()