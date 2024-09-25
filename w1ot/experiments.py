from .pytorch_implementation import w1ot, w2ot
import os
import anndata as ad
import pandas as pd
import pickle
import numpy as np
from typing import Any, Dict
from anndata import AnnData

class Experiment:
    def __init__(self, 
                 model_name: str, 
                 dataset: AnnData, 
                 exp_type: str,
                 embedding: bool,
                 latent_dim: int,
                 output_dir: str,
                 test_size: float = 0.1,
                 device: str = None) -> None:
        """
        Initialize the Experiment class.

        Parameters:
            model_name (str): Name of the model to use. Options are 'w1ot', 'w2ot'.
            dataset (AnnData): The anndata dataset containing observations with annotations.
            exp_type (str): Type of experiment to run. Options are 'iid', 'out_of_dosage', 'out_of_celltype'.
            embedding (bool): Whether to use the embedding or not.
            latent_dim (int): Dimension of the latent space for the embedding.
            output_dir (str): Directory where results and models will be saved.
            test_size (float, optional): Proportion of the dataset to use for testing. Defaults to 0.1.
            device (str, optional): Computing device to use. If None, uses CUDA if available, else CPU. Defaults to None.

        Example:
            >>> import anndata
            >>> adata = anndata.read_h5ad("my_dataset.h5ad")
            >>> experiment = Experiment(model_name='w1ot',
            ...                         dataset=adata,
            ...                         exp_type='iid',
            ...                         embedding=True,
            ...                         latent_dim=32,
            ...                         output_dir='./results',
            ...                         test_size=0.2,
            ...                         device='cuda')
        """
        self.model_name = model_name
        if model_name == 'w1ot':
            self.model = w1ot
        elif model_name == 'w2ot':
            self.model = w2ot
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        self.dataset = dataset
        self.exp_type = exp_type
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories for models and results
        self.model_dir = os.path.join(self.output_dir, self.model_name)
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.results: Dict[str, Dict[str, Any]] = {}
        self.test_size = test_size
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding = embedding
        self.latent_dim = latent_dim
    
    def _train_embedding(self, 
                         adata: AnnData,
                         path: str) -> None:
        """
        Embed the data using scvi-tools' scVI model.

        Parameters:
            adata (AnnData): The AnnData object containing the dataset to embed.

        Returns:
            None

        Note:
            This method trains the VAE model and stores it in self.embedding_model.
        """
        self.embedding_model = VAE(device='cpu', output_dir=path)
        self.embedding_model.setup_anndata(adata)
        self.embedding_model.setup_model(hidden_layers=[128, 128], latent_dim=self.latent_dim)
        self.embedding_model.train(num_iters=100000, batch_size=128, lr=1e-4, resume_from_checkpoint=True, checkpoint_interval=1000)
    
    def _get_embedding(self, 
                       data: Any) -> np.ndarray:
        """
        Get the latent embeddings of the data.

        Parameters:
            data (Any): The data to embed. Can be an AnnData object or a numpy array.

        Returns:
            np.ndarray: The latent representation of the input data.

        Note:
            This method uses the trained VAE model to generate embeddings.
        """
        if isinstance(data, AnnData):
            data = self._ensure_numpy(data.X)
            data = self.embedding_model.normalize_and_log_transform(data)
            return self.embedding_model.get_latent_representation(data)
        else:
            data = self._ensure_numpy(data)
            data = self.embedding_model.normalize_and_log_transform(data)
            return self.embedding_model.get_latent_representation(data)
    
    def _decode_embedding(self, 
                          embedding: np.ndarray) -> np.ndarray:
        """
        Decode the latent embeddings back to the original data space.

        Parameters:
            embedding (np.ndarray): The latent embeddings to decode.

        Returns:
            np.ndarray: The decoded data in the original space.

        Note:
            This method uses the trained VAE model to decode the embeddings.
        """
        embedding = torch.FloatTensor(embedding).to(self.device)

        return self.embedding_model.decode(embedding)


    def _ensure_numpy(self, data: Any) -> np.ndarray:
        """
        Ensure the data is a NumPy ndarray.

        Parameters:
            data (Any): The data to be converted.

        Returns:
            np.ndarray: Converted NumPy array.

        Note:
            This method handles different input types and converts them to numpy arrays.
        """
        if isinstance(data, np.ndarray):
            return data
        elif hasattr(data, 'toarray'):
            return data.toarray()
        else:
            return np.array(data)

    def run(self, train_category: str) -> None:
        """
        Run the experiment based on the specified type.

        Parameters:
            train_category (str): The category to use for training in out-of-distribution experiments.

        Returns:
            None

        Note:
            This method dispatches to the appropriate experiment type based on self.exp_type.
        """
        if self.exp_type == 'iid':
            self.run_iid_perturbation()
        elif self.exp_type == 'out_of_dosage':
            self.run_out_of_distribution('dosage', train_category)
        elif self.exp_type == 'out_of_celltype':
            self.run_out_of_distribution('celltype', train_category)
        else:
            raise ValueError(f"Unknown experiment type: {self.exp_type}")

    def run_iid_perturbation(self) -> None:
        """
        Run IID perturbation experiments.

        Returns:
            None

        Note:
            This method runs the IID perturbation experiment for each unique perturbation in the dataset.
        """
        perturbations = self.dataset.obs['perturbation'].unique()

        for perturbation in perturbations:
            if perturbation == 'control':
                continue
            
            source_adata = self.dataset[self.dataset.obs['perturbation'] == 'control'].copy()
            target_adata = self.dataset[self.dataset.obs['perturbation'] == perturbation].copy()

            # Save the trained model
            model_path = os.path.join(self.model_dir, f'{perturbation}')

            # split the data into train and test
            source_test_size = int(len(source_adata) * self.test_size)
            target_test_size = int(len(target_adata) * self.test_size)
            source_train_adata = source_adata[:-source_test_size]
            source_test_adata = source_adata[-source_test_size:]
            target_train_adata = target_adata[:-target_test_size]
            target_test_adata = target_adata[-target_test_size:]

            # embed the data
            if self.embedding:
                self._train_embedding(ad.concat([source_train_adata, target_train_adata]), model_path)
                source_train = self._get_embedding(source_train_adata)
                source_train_adata.obsm['X_emb'] = source_train
                target_train = self._get_embedding(target_train_adata)
                target_train_adata.obsm['X_emb'] = target_train
                source_test = self._get_embedding(source_test_adata)
                source_test_adata.obsm['X_emb'] = source_test
                target_test = self._get_embedding(target_test_adata)
                target_test_adata.obsm['X_emb'] = target_test
            else:
                source_train = self._ensure_numpy(source_train_adata.X)
                target_train = self._ensure_numpy(target_train_adata.X)
                source_test = self._ensure_numpy(source_test_adata.X)
                target_test = self._ensure_numpy(target_test_adata.X)

            # Initialize and train w1ot model
            model = self.model(source=source_train, target=target_train, device=self.device, path=model_path)
            model.fit_potential_function()
            if self.model_name == 'w1ot':
                model.fit_distance_function()

            model.save(model_path)

            # Predict on test data
            transported = model.transport(source_test)
            
            # transform the transported data back to the original space
            if self.embedding:
                transported_adata = self._decode_embedding(transported)
                transported_adata = AnnData(transported_adata, obs=source_test_adata.obs, var=source_test_adata.var)
                transported_adata.obsm['X_emb'] = transported
            else:
                transported_adata = AnnData(transported, obs=source_test_adata.obs, var=source_test_adata.var)
            

            # Save results as anndata
            source_train_adata.write(os.path.join(model_path, f'source_train_{perturbation}.h5ad'))
            target_train_adata.write(os.path.join(model_path, f'target_train_{perturbation}.h5ad'))
            transported_adata.write(os.path.join(model_path, f'transported_{perturbation}.h5ad'))
            source_test_adata.write(os.path.join(model_path, f'source_test_{perturbation}.h5ad'))
            target_test_adata.write(os.path.join(model_path, f'target_test_{perturbation}.h5ad'))
            

    def run_out_of_distribution(self, ood_type: str, train_category: str) -> None:
        """
        Run out-of-distribution perturbation experiments.

        Parameters:
            ood_type (str): Type of out-of-distribution experiment to run. Options are 'dosage', 'celltype'.
            train_category (str): The category to use for training.

        Returns:
            None

        Note:
            This method runs the out-of-distribution experiment for each unique perturbation in the dataset,
            training on one category and testing on all others.
        """
        if ood_type == 'dosage':
            categories = self.dataset.obs['dosage'].unique()
        elif ood_type == 'celltype':
            categories = self.dataset.obs['celltype'].unique()
        else:
            raise ValueError(f"Unknown type: {ood_type}")

        perturbations = self.dataset.obs['perturbation'].unique()
        # exclude control in perturbation list
        perturbations = [perturbation for perturbation in perturbations if perturbation != 'control']

        for perturbation in perturbations:
            # Train data: control and perturbed data for the first category
            source_train_adata = self.dataset[
                (self.dataset.obs[ood_type] == train_category) &
                (self.dataset.obs['perturbation'] == 'control')
            ].copy()

            target_train_adata = self.dataset[
                (self.dataset.obs[ood_type] == train_category) &
                (self.dataset.obs['perturbation'] == perturbation)
            ].copy()

            # Save the trained model
            ood_type_dir = os.path.join(self.model_dir, ood_type)
            os.makedirs(ood_type_dir, exist_ok=True)

            model_path = os.path.join(ood_type_dir, f'{perturbation}_{train_category}')
            os.makedirs(model_path, exist_ok=True)

            # embed the training data
            if self.embedding:
                self._train_embedding(ad.concat([source_train_adata, target_train_adata]), model_path)
                source_train = self._get_embedding(source_train_adata)
                source_train_adata.obsm['X_emb'] = source_train
                target_train = self._get_embedding(target_train_adata)
                target_train_adata.obsm['X_emb'] = target_train
            else:
                source_train = self._ensure_numpy(source_train_adata.X)
                target_train = self._ensure_numpy(target_train_adata.X)

            # Initialize and train w1ot model
            model = self.model(source=source_train, target=target_train, device=self.device, path=model_path)
            model.fit_potential_function()
            if self.model_name == 'w1ot':
                model.fit_distance_function()

            # Save the trained model
            model.save(model_path)

            # Test on all other categories
            for test_category in categories:
                if test_category == train_category:
                    continue

                source_test_adata = self.dataset[
                    (self.dataset.obs[ood_type] == test_category) &
                    (self.dataset.obs['perturbation'] == 'control')
                ].copy()

                target_test_adata = self.dataset[
                    (self.dataset.obs[ood_type] == test_category) &
                    (self.dataset.obs['perturbation'] == perturbation)
                ].copy()

                # embed the test data
                if self.embedding:
                    source_test = self._get_embedding(source_test_adata)
                    source_test_adata.obsm['X_emb'] = source_test
                    target_test = self._get_embedding(target_test_adata)
                    target_test_adata.obsm['X_emb'] = target_test
                else:
                    source_test = self._ensure_numpy(source_test_adata.X)
                    target_test = self._ensure_numpy(target_test_adata.X)

                # Predict on test data
                transported = model.transport(source_test)
                
                # transform the transported data back to the original space
                if self.embedding:
                    transported_adata = self._decode_embedding(transported)
                    transported_adata = AnnData(transported_adata, obs=source_test_adata.obs, var=source_test_adata.var)
                    transported_adata.obsm['X_emb'] = transported
                else:
                    transported_adata = AnnData(transported, obs=source_test_adata.obs, var=source_test_adata.var)
                
                # Save results as anndata
                source_test_adata.write(os.path.join(model_path, f'source_test_{ood_type}_{train_category}_{test_category}_{perturbation}.h5ad'))
                target_test_adata.write(os.path.join(model_path, f'target_test_{ood_type}_{train_category}_{test_category}_{perturbation}.h5ad'))
                transported_adata.write(os.path.join(model_path, f'transported_{ood_type}_{train_category}_{test_category}_{perturbation}.h5ad'))

            # Save training data
            source_train_adata.write(os.path.join(model_path, f'source_train_{ood_type}_{train_category}_{perturbation}.h5ad'))
            target_train_adata.write(os.path.join(model_path, f'target_train_{ood_type}_{train_category}_{perturbation}.h5ad'))


import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import scanpy as sc

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
              validation_split: float = 0.1,
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
        val_size = int(validation_split * dataset_size)
        train_size = dataset_size - val_size
        self.train_data, self.val_data = torch.split(self.data, [train_size, val_size])
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
            self.model.eval()
            with torch.no_grad():
                val_batch = self._sample_data(self.val_data, batch_size)
                val_recon, val_mu, val_logvar = self.model(val_batch)
                val_loss, val_recon, val_kld = self.model.loss_function(val_recon, val_batch, val_mu, val_logvar, alpha)
            
            
            progress_bar.set_postfix({'ELBO': val_loss.item()})
            
            # save checkpoint periodically
            if (iteration + 1) % checkpoint_interval == 0 and val_loss < self.best_val_elbo:
                self.best_val_elbo = val_loss
                self._save_checkpoint(iteration + 1, optimizer, checkpoint_path)
        
        print(f'Training completed. Best Val ELBO: {self.best_val_elbo:.4f}')
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
