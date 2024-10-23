import os
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from typing import Dict, Any, List, Tuple

from .models import LBNN, DNN, ICNN

def reproducibility(seed=1):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


class OTDataset:
    def __init__(self, 
                 source,
                 target,
                 transport_direction = None,
                 device = "cpu"):

        self.source = torch.FloatTensor(source).to(device)
        self.target = torch.FloatTensor(target).to(device)
        if transport_direction is not None:
            self.transport_direction = torch.FloatTensor(transport_direction).to(device)
        else:
            self.transport_direction = None
        self.p = source.shape[1]
        self.device = device

        # Initialize permutation indices and counters for source and target
        self.source_indices = torch.randperm(self.source.size(0))
        self.source_idx = 0

        self.target_indices = torch.randperm(self.target.size(0))
        self.target_idx = 0

    def sample_target(self, batch_size: int):
        if batch_size > self.target.size(0):
            batch_size = self.target.size(0)
        
        if self.target_idx + batch_size > self.target.size(0):
            # Reshuffle and reset index
            self.target_indices = torch.randperm(self.target.size(0))
            self.target_idx = 0

        idx = self.target_indices[self.target_idx:self.target_idx + batch_size]
        self.target_idx += batch_size
        return self.target[idx]

    def sample_source(self, batch_size: int):
        if batch_size > self.source.size(0):
            batch_size = self.source.size(0)
        
        if self.source_idx + batch_size > self.source.size(0):
            # Reshuffle and reset index
            self.source_indices = torch.randperm(self.source.size(0))
            self.source_idx = 0

        idx = self.source_indices[self.source_idx:self.source_idx + batch_size]
        self.source_idx += batch_size
        if self.transport_direction is not None:
            return self.source[idx], self.transport_direction[idx]
        else:
            return self.source[idx]

class w1ot:
    def __init__(self,
                 source: np.ndarray = None,
                 target: np.ndarray = None,
                 positive: bool = False,
                 validation_size: float = 0.1,
                 device: str = None,
                 path: str = None):
        """
        Initialize the Wasserstein-1 optimal transport model.

        Parameters:
        source (np.ndarray): Data sampled from the source distribution.
        target (np.ndarray): Data sampled from the target distribution.
        positive (bool): Whether to use positive transport direction. Default is False.
        validation_size (float): Validation size, between 0 and 1. Default is 0.1.
        device (str): Computing device of PyTorch backend. Default is None.
        path (str): Path to save the model. Default is None.
        """
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if path is not None:
            self.path = path
        else:
            self.path = './saved_models/w1ot'
        
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if source is not None and target is not None:
            # Split the dataset into training and validation sets
            
            val_size = int(validation_size * min(source.shape[0], target.shape[0]))
            train_size_source = source.shape[0] - val_size
            train_size_target = target.shape[0] - val_size

            self.dataset = OTDataset(source[:train_size_source], target[:train_size_target], device=self.device)
            self.val_dataset = OTDataset(source[train_size_source:], target[train_size_target:], device=self.device)
            self.p = source.shape[1]
        else:
            self.dataset = None
            self.val_dataset = None

        self.network_config = {}
        # Kantorovich potential
        self.phi_network_opt = {}
        self.phi = None

        # step size function
        self.eta_network_opt = {}
        self.eta = None

        # discriminator
        self.d_network_opt = {}
        self.D = None

        self.positive = positive

    def fit_potential_function(self,
                           hidden_sizes: List[int] = [64]*4,
                           orthornormal_layer: str = 'cayley',
                           groups: int = 4,
                           batch_size: int = 256,
                           num_iters: int = 10000,
                           lr_init: float = 1e-2,
                           lr_min: float = 1e-4,
                           betas: Tuple[float, float] = (0.5, 0.5),
                           checkpoint_interval: int = 100,
                           resume_from_checkpoint: bool = False) -> None:
        """
        Fit the potential function using the specified parameters.

        Parameters:
        hidden_sizes (List[int]): List of hidden layer sizes. Default is [64]*4.
        orthornormal_layer (str): Type of orthonormal layer to use. Default is 'cayley'.
        groups (int): Number of groups for GroupSort. Default is 4.
        batch_size (int): Batch size for training. Default is 256.
        num_iters (int): Number of iterations for training. Default is 10000.
        lr_init (float): Initial learning rate. Default is 1e-2.
        lr_min (float): Minimum learning rate. Default is 1e-4.
        betas (Tuple[float, float]): Betas for the Adam optimizer. Default is (0.5, 0.5).
        checkpoint_interval (int): Interval for saving checkpoints. Default is 100.
        resume_from_checkpoint (bool): Whether to resume from a checkpoint. Default is False.
        """
        reproducibility()
        self.phi_network_opt['input_size'] = self.p
        self.phi_network_opt['output_size'] = 1
        self.phi_network_opt['scale'] = 1
        self.phi_network_opt['hidden_sizes'] = hidden_sizes
        self.phi_network_opt['orthornormal_layer'] = orthornormal_layer
        self.phi_network_opt['groups'] = groups
        self.network_config['phi'] = self.phi_network_opt
        self.phi = LBNN(**self.phi_network_opt).to(self.device)

        # Optimizer
        optimizer_phi = optim.Adam(self.phi.parameters(), betas=betas, lr=lr_init, maximize=True)
        # Initialize the cosine annealing scheduler
        scheduler = CosineAnnealingLR(optimizer_phi, T_max=num_iters, eta_min=lr_min)

        start_iter = 0
        
        if resume_from_checkpoint:
            try:
                checkpoint = self._load_checkpoint('potential')
                self.phi.load_state_dict(checkpoint['model_state_dict'])
                optimizer_phi.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_iter = checkpoint['iteration']
                best_val_w1 = checkpoint['best_val_w1']
                print(f"Resuming from iteration {start_iter} with best validation W1: {best_val_w1}")
            except:
                print(f"Could not load checkpoint. Training from scratch.")

        if start_iter > 0:
            for _ in range(start_iter):
                scheduler.step()
        
        self.phi.train()

        best_val_w1 = 0

        progress_bar = tqdm(range(start_iter, num_iters), desc="Training potential")
        for iteration in progress_bar:
            source_batch, target_batch = self.dataset.sample_source(batch_size), self.dataset.sample_target(batch_size)

            optimizer_phi.zero_grad()
            loss = (self.phi(source_batch).mean() - self.phi(target_batch).mean())
            loss.backward()  
            optimizer_phi.step()
            train_loss = loss.item()
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            progress_bar.set_postfix({
                'W1': f'{train_loss:.2f}',
            })

            if (iteration + 1) % checkpoint_interval == 0:
                if self.val_dataset is not None:
                    self.phi.eval()
                    with torch.no_grad():
                        val_source_batch, val_target_batch = self.val_dataset.sample_source(10e+8), self.val_dataset.sample_target(10e+8)
                        val_w1 = (self.phi(val_source_batch).mean() - self.phi(val_target_batch).mean()).item()
                    self.phi.train()
                    if val_w1 > best_val_w1:
                        best_val_w1 = val_w1
                
                self._save_checkpoint('potential', iteration + 1, self.phi, optimizer_phi, scheduler)

    def fit_distance_function(self,
                              d_hidden_sizes: List[int] = [64]*4,
                              eta_hidden_sizes: List[int] = [64]*4,
                              batch_size: int = 256,
                              num_iters: int = 10000,
                              lr_init: float = 1e-4,
                              lr_min: float = 1e-4,
                              betas: Tuple[float, float] = (0.9, 0.999),
                              checkpoint_interval: int = 100,
                              resume_from_checkpoint: bool = False) -> None:
        """
        Fit the distance function using the specified parameters.

        Parameters:
        d_hidden_sizes (List[int]): List of hidden layer sizes for the discriminator. Default is [64]*4.
        eta_hidden_sizes (List[int]): List of hidden layer sizes for the eta network. Default is [64]*4.
        batch_size (int): Batch size for training. Default is 256.
        num_iters (int): Number of iterations for training. Default is 10000.
        lr_init (float): Initial learning rate. Default is 1e-4.
        lr_min (float): Minimum learning rate. Default is 1e-4.
        betas (Tuple[float, float]): Betas for the Adam optimizer. Default is (0.9, 0.999).
        checkpoint_interval (int): Interval for saving checkpoints. Default is 100.
        resume_from_checkpoint (bool): Whether to resume from a checkpoint. Default is False.
        """
        reproducibility()
        self.eta_network_opt['input_size'] = self.p
        self.eta_network_opt['output_size'] = 1
        self.eta_network_opt['hidden_sizes'] = eta_hidden_sizes
        self.eta_network_opt['final_activation'] = 'softplus'
        self.eta = DNN(**self.eta_network_opt).to(self.device)
        reproducibility()
        self.d_network_opt['input_size'] = self.p
        self.d_network_opt['output_size'] = 1
        self.d_network_opt['hidden_sizes'] = d_hidden_sizes
        self.d_network_opt['final_activation'] = 'sigmoid'
        self.D = DNN(**self.d_network_opt).to(self.device)

        self.network_config['eta'] = self.eta_network_opt
        self.network_config['D'] = self.d_network_opt
       
        optimizer_D = optim.Adam(self.D.parameters(), lr=lr_init, betas=betas)
        optimizer_eta = optim.Adam(self.eta.parameters(), lr=lr_init, betas=betas)
        

        scheduler_D = CosineAnnealingLR(optimizer_D, T_max=num_iters, eta_min=lr_min)
        scheduler_eta = CosineAnnealingLR(optimizer_eta, T_max=num_iters, eta_min=lr_min)

        start_iter = 0
        if resume_from_checkpoint:
            try:
                checkpoint = self._load_checkpoint('distance')
                self.eta.load_state_dict(checkpoint['eta_state_dict'])
                self.D.load_state_dict(checkpoint['D_state_dict'])
                optimizer_eta.load_state_dict(checkpoint['optimizer_eta_state_dict'])
                optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
                scheduler_eta.load_state_dict(checkpoint['scheduler_eta_state_dict'])
                scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
                start_iter = checkpoint['iteration']
                print(f"Resuming from iteration {start_iter}")
            except:
                print(f"Could not load checkpoint. Training from scratch.")

        self.eta.train()
        self.D.train()

        progress_bar = tqdm(range(start_iter, num_iters), desc="Training")\

        checkpoint = self._load_checkpoint('potential')
        self.phi.load_state_dict(checkpoint['model_state_dict'])

        source = self.dataset.source.requires_grad_(True)
        (direction,) = torch.autograd.grad(
            self.phi(source),
            source,
            create_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((source.size()[0], 1), device=source.device).float(),
        )
        self.dataset.transport_direction = direction.detach()/(direction.detach().norm(p=2, dim=1).reshape(-1,1) + 1e-6)
        self.dataset.source.requires_grad_(False)

        for iteration in progress_bar:
            source_batch, direction_batch = self.dataset.sample_source(batch_size)
            target_batch = self.dataset.sample_target(batch_size)
            
            optimizer_eta.zero_grad()
            optimizer_D.zero_grad()

            # Transport the source batch using the current alpha network
            if self.positive:
                transported_batch = F.relu(source_batch - self.eta(source_batch) * direction_batch)
            else:
                transported_batch = source_batch - self.eta(source_batch) * direction_batch

            # Discriminator loss: Use log loss
            D_real = torch.mean(self.D(target_batch))
            discriminator_loss_1 = - torch.mean(torch.log(self.D(target_batch) + 1e-6))
            discriminator_loss_1.backward()

            discriminator_loss_2 = - torch.mean(torch.log(1 - self.D(transported_batch.detach()) + 1e-6))
            # Update the discriminator
            discriminator_loss_2.backward()
            optimizer_D.step()

            # Generator (Transport) loss: Use log loss to fool the discriminator
            D_fake = self.D(transported_batch)  # Recompute D_fake for the generator
            generator_loss = - torch.mean(torch.log(D_fake + 1e-6))
            
            # Update the distance function
            generator_loss.backward()
            optimizer_eta.step()

            scheduler_D.step()
            scheduler_eta.step()

            progress_bar.set_postfix({
                'D': f'{D_real.item():.2f}',
                'G': f'{torch.mean(D_fake).item():.2f}',
            })
                
            if (iteration + 1) % checkpoint_interval == 0:
                self._save_checkpoint('distance', iteration + 1, self.eta, optimizer_eta, scheduler_eta, 
                                      self.D, optimizer_D, scheduler_D)

        print("Final GAN stats:")
        print("D_real: ", D_real.item())
        print("D_fake: ", torch.mean(D_fake).item())
        self._save_model(self.path)



    def transport(self,
                  source: np.ndarray,
                  method: str = 'GAN') -> np.ndarray:
        """
        Transport the source data to the target data using the trained transport map.

        Parameters:
        source (np.ndarray): Data points in the source distribution.
        method (str): Method to use for transport. Default is 'GAN'.

        Returns:
        np.ndarray: Transported source data under Wasserstein-1 optimal transport.
        """
        if method == 'GAN':
            return self._transport_GAN(source)


    def _transport_GAN(self, source: np.ndarray) -> np.ndarray:
        """
        Transport the source data to the target data using the gradient guidance method.
        The transported sample is computed as x - eta(x) * grad(phi(x)), where eta(x) is the distance function.

        Parameters:
        source (np.ndarray): Data points in the source distribution.

        Returns:
        np.ndarray: Transported source data.
        """
        self.eta.eval()
        self.phi.eval()
        transported = []

        source = torch.tensor(source, dtype=torch.float32, device=self.device).requires_grad_(True)

        (direction,) = torch.autograd.grad(
            self.phi(source),
            source,
            create_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((source.size()[0], 1), device=source.device).float(),
        )

        print("Min Lipschitz constant: ", direction.norm(p=2, dim=1).min().item())
        print("Max Lipschitz constant: ", direction.norm(p=2, dim=1).max().item())
        print("Mean Lipschitz constant: ", direction.norm(p=2, dim=1).mean().item())

        if self.positive:
            transported = F.relu(source - self.eta(source) * direction)
        else:
            transported = source - self.eta(source) * direction

        return transported.cpu().detach().numpy()

    def _save_model(self, path: str) -> None:
        """
        Save the model to the specified path.

        Parameters:
        path (str): Path to save the model.
        """
        if not os.path.exists(path):
            os.makedirs(path)

        # save phi
        if self.phi is not None:
            self.network_config['phi_state_dict'] = self.phi.state_dict()
        else:
            self.network_config['phi'] = None
            self.network_config['phi_state_dict'] = None
        # save eta
        if self.eta is not None:
            self.network_config['eta_state_dict'] = self.eta.state_dict()
        else:
            self.network_config['eta'] = None
            self.network_config['eta_state_dict'] = None
        # save D
        if self.D is not None:
            self.network_config['D_state_dict'] = self.D.state_dict()
        else:
            self.network_config['D'] = None
            self.network_config['D_state_dict'] = None
        
        torch.save(self.network_config, os.path.join(path, 'w1ot_networks.pt'))

    def _save_checkpoint(self, checkpoint_type, iteration, *args):
        checkpoint = {
            'iteration': iteration,
        }
        if checkpoint_type == 'potential':
            model, optimizer, scheduler = args
            checkpoint.update({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            })
        elif checkpoint_type == 'distance':
            eta, optimizer_eta, scheduler_eta, D, optimizer_D, scheduler_D = args
            checkpoint.update({
                'eta_state_dict': eta.state_dict(),
                'optimizer_eta_state_dict': optimizer_eta.state_dict(),
                'scheduler_eta_state_dict': scheduler_eta.state_dict(),
                'D_state_dict': D.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'scheduler_D_state_dict': scheduler_D.state_dict(),
            })
        
        torch.save(checkpoint, os.path.join(self.path, f'{checkpoint_type}_checkpoint.pt'))

    def _load_checkpoint(self, checkpoint_type):
        checkpoint_path = os.path.join(self.path, f'{checkpoint_type}_checkpoint.pt')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        print("Checkpoint loaded.")
        return checkpoint

    def load(self, path):
        self.network_config = torch.load(path)
        # load phi
        if self.network_config['phi'] is not None:
            self.phi = LBNN(**self.network_config['phi'])
            self.phi.load_state_dict(self.network_config['phi_state_dict'])
            self.phi = self.phi.to(self.device)
        
        if self.network_config['eta'] is not None:
            self.eta = DNN(**self.network_config['eta'])
            self.eta.load_state_dict(self.network_config['eta_state_dict'])
            self.eta = self.eta.to(self.device)
        if self.network_config['D'] is not None:
            self.D = DNN(**self.network_config['D'])
            self.D.load_state_dict(self.network_config['D_state_dict'])
            self.D = self.D.to(self.device)
    
    def save(self, path):
        self._save_model(path)

    def plot_2dpotential(self, resolution: int = 100) -> None:
        """
        Plot the 2D potential map.

        This method visualizes the learned potential function in 2D space. It creates a contour plot
        of the potential values over a grid of points, and overlays the source and target data points.

        Parameters:
        resolution (int): The number of points along each axis in the plot grid. Default is 100.

        Returns:
        None: This method doesn't return anything, it displays the plot.

        Raises:
        ValueError: If the input data is not 2-dimensional.
        RuntimeError: If the phi network is not initialized.

        Note:
        This method requires matplotlib to be installed.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if self.dataset.p != 2:
            raise ValueError("This method only works for 2D data.")

        if self.phi is None:
            raise RuntimeError("The potential function (phi) is not initialized. Train the model first.")

        # Make the plot range suitable with source data and target data
        x_range = (min(self.dataset.source[:, 0].min(), self.dataset.target[:, 0].min()) - 0.1,
                   max(self.dataset.source[:, 0].max(), self.dataset.target[:, 0].max()) + 0.1)
        y_range = (min(self.dataset.source[:, 1].min(), self.dataset.target[:, 1].min()) - 0.1,
                   max(self.dataset.source[:, 1].max(), self.dataset.target[:, 1].max()) + 0.1)

        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        grid_points = np.column_stack((X.ravel(), Y.ravel()))
        grid_points_tensor = torch.FloatTensor(grid_points).to(self.device)

        self.phi.eval()
        with torch.no_grad():
            Z = self.phi(grid_points_tensor).cpu().numpy().reshape(X.shape)

        plt.figure(figsize=(10, 8))
        contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
        plt.colorbar(contour, label='Potential')

        contour_lines = plt.contour(X, Y, Z, levels=20, colors='k', alpha=0.5, linewidths=0.5)
        plt.clabel(contour_lines, inline=True, fontsize=8)

        plt.title('2D Potential Map')
        plt.xlabel('X')
        plt.ylabel('Y')

        # Plot source and target points
        plt.scatter(self.dataset.source[:, 0].cpu(), self.dataset.source[:, 1].cpu(), c='b', s=10,
                    label='Source')
        plt.scatter(self.dataset.target[:, 0].cpu(), self.dataset.target[:, 1].cpu(), c='r', s=10,
                    label='Target')
        plt.legend(loc='upper right')

        plt.show()



class w2ot:
    def __init__(self,
                 source: np.ndarray = None,
                 target: np.ndarray = None,
                 positive: bool = False,
                 validation_size: float = 0.1,
                 device: str = None,
                 path: str = None,
                 ) -> None:
        """
        Initialize the Wasserstein-2 optimal transport model.

        Parameters:
        source (np.ndarray): Data sampled from the source distribution.
        target (np.ndarray): Data sampled from the target distribution.
        positive (bool): Whether to use positive transport direction. Default is False.
        validation_size (float): Validation size, between 0 and 1. Default is 0.1.
        device (str): Computing device of PyTorch backend. Default is None.
        path (str): Path to save the model. Default is None.
        """
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if path is not None:
            self.path = path
        else:
            self.path = './saved_models/w2ot'
        
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if source is not None and target is not None:
            # Split the dataset into training and validation sets
            val_size = int(validation_size * min(source.shape[0], target.shape[0]))
            train_size_source = source.shape[0] - val_size
            train_size_target = target.shape[0] - val_size

            self.dataset = OTDataset(source[:train_size_source], target[:train_size_target], device=self.device)
            self.val_dataset = OTDataset(source[train_size_source:], target[train_size_target:], device=self.device)
            self.p = source.shape[1]
        else:
            self.dataset = None
            self.val_dataset = None
        
        self.f = None
        self.g = None

        self.scheduler_f = None
        self.scheduler_g = None

        self.network_config = {}
        self.positive = positive
    
    def fit_potential_function(self,
                               flavor: str = 'ot_icnn',
                               hidden_sizes: List[int] = [64]*4,
                               batch_size: int = 256,
                               num_iters: int = 100000,
                               num_inner_iter: int = 10,
                               lr_init: float = 1e-3,
                               lr_min: float = 1e-4,
                               betas: Tuple[float, float] = (0.5, 0.9),
                               kernel_init: str = None,
                               resume_from_checkpoint: bool = False,
                               checkpoint_interval: int = 100,
                               **kwargs) -> None:
        """
        Train the potential functions f and g.

        Parameters:
        flavor (str): Type of network architecture to use. Options: 'cellot', 'ot_icnn', 'custom'. Default is 'ot_icnn'.
        hidden_sizes (List[int]): List of hidden layer sizes. Default is [64,64,64,64].
        batch_size (int): Batch size for training. Default is 256.
        num_iters (int): Number of training iterations. Default is 100000.
        num_inner_iter (int): Number of inner iterations for g updates. Default is 10.
        lr_init (float): Initial learning rate. Default is 1e-3.
        lr_min (float): Minimum learning rate for scheduler. Default is 1e-4.
        betas (Tuple[float, float]): Adam optimizer betas. Default is (0.5, 0.9).
        kernel_init (str): Kernel initialization method. Default is None.
        resume_from_checkpoint (bool): Whether to resume from checkpoint. Default is False.
        checkpoint_interval (int): Interval for saving checkpoints. Default is 100.
        **kwargs: Additional arguments.
        """

        # Initialize the ICNN
        if flavor == 'cellot':
            self.network_config['f'] = {'input_size': self.p, 'hidden_sizes': [64]*4, 'fnorm_penalty': 0, 'kernel_init': 'uniform', 'b': 0.1}
            self.network_config['g'] = {'input_size': self.p, 'hidden_sizes': [64]*4, 'fnorm_penalty': 1, 'kernel_init': 'uniform', 'b': 0.1}
            self.f = ICNN(**self.network_config['f']).to(self.device)
            self.g = ICNN(**self.network_config['g']).to(self.device)
        
            # Initialize the optimizers
            self.optimizer_f = optim.Adam(self.f.parameters(), lr=1e-4, betas=(0.5, 0.9))
            self.optimizer_g = optim.Adam(self.g.parameters(), lr=1e-4, betas=(0.5, 0.9))
            batch_size = 256
            num_iters = 100000

        elif flavor == 'ot_icnn':
            self.network_config['f'] = {'input_size': self.p, 'hidden_sizes': [64]*4, 'fnorm_penalty': 0, 'kernel_init': None}
            self.network_config['g'] = {'input_size': self.p, 'hidden_sizes': [64]*4, 'fnorm_penalty': 1, 'kernel_init': None}
            self.f = ICNN(**self.network_config['f']).to(self.device)
            self.g = ICNN(**self.network_config['g']).to(self.device)
        
            # Initialize the optimizers
            self.optimizer_f = optim.Adam(self.f.parameters(), lr=1e-4, betas=(0.5, 0.9))
            self.optimizer_g = optim.Adam(self.g.parameters(), lr=1e-4, betas=(0.5, 0.9))
            batch_size = 256
            num_iters = 100000

        elif flavor == 'custom':
            self.network_config['f'] = {'input_size': self.p, 'hidden_sizes': hidden_sizes, 'fnorm_penalty': 0, 'kernel_init':kernel_init}
            self.network_config['g'] = {'input_size': self.p, 'hidden_sizes': hidden_sizes, 'fnorm_penalty': 1, 'kernel_init':kernel_init}
            self.f = ICNN(**self.network_config['f']).to(self.device)
            self.g = ICNN(**self.network_config['g']).to(self.device)
        
            self.optimizer_f = optim.Adam(self.f.parameters(), lr=lr_init, betas=betas)
            self.optimizer_g = optim.Adam(self.g.parameters(), lr=lr_init, betas=betas)
            self.scheduler_f = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_f, T_max=num_iters, eta_min=lr_min)
            self.scheduler_g = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_g, T_max=num_iters, eta_min=lr_min)

        else:
            raise ValueError(f"Invalid flavor: {flavor}")

        start_iter = 0
        if resume_from_checkpoint:
            try:
                checkpoint = self._load_checkpoint()
                self.f.load_state_dict(checkpoint['f_state_dict'])
                self.g.load_state_dict(checkpoint['g_state_dict'])
                self.optimizer_f.load_state_dict(checkpoint['optimizer_f_state_dict'])
                self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
                start_iter = checkpoint['iteration']
                best_val_w2 = checkpoint['best_val_w2']
                print(f"Resuming from iteration {start_iter} with best validation W2: {best_val_w2}")
            except:
                print(f"Could not load checkpoint. Starting training from scratch.")
        
        # Optimization loop
        progress_bar = tqdm(range(start_iter, num_iters), desc="Training")
        for iteration in progress_bar:
            
            # Get the target batch
            target_batch = self.dataset.sample_target(batch_size)
            
            for _ in range(num_inner_iter):
                # Sample batch from source
                source_batch = self.dataset.sample_source(batch_size)
                source_batch.requires_grad_(True)
                # Compute loss for g
                self.optimizer_g.zero_grad()
                loss_g = self._compute_loss_g(self.f, self.g, source_batch).mean()
                
                if self.g.fnorm_penalty > 0:
                    loss_g += self.g.penalize_w()

                loss_g.backward()
                self.optimizer_g.step()
            
            # Sample batches for f update
            source_batch = self.dataset.sample_source(batch_size)
            source_batch.requires_grad_(True)
            
            # Compute loss for f
            self.optimizer_f.zero_grad()
            loss_f = self._compute_loss_f(self.f, self.g, source_batch, target_batch).mean()
            loss_f.backward()
            self.optimizer_f.step()
        
            self.f.clamp_w()

            if self.scheduler_f is not None and self.scheduler_g is not None:
                self.scheduler_f.step()
                self.scheduler_g.step()

            w2_distance = self.compute_w2_distance(self.f, self.g, source_batch, target_batch)

            progress_bar.set_postfix({
                'W2': f'{w2_distance.item():.2f}',
            })

            if (iteration + 1) % checkpoint_interval == 0:
                self.f.eval()
                self.g.eval()
                val_source_batch, val_target_batch = self.val_dataset.sample_source(10e+8), self.val_dataset.sample_target(10e+8)
                val_source_batch.requires_grad_(True)
                val_w2 = self.compute_w2_distance(self.f, self.g, val_source_batch, val_target_batch).item()
                self.f.train()
                self.g.train()
                self._save_checkpoint(iteration + 1, val_w2)
        self._save_model(self.path)
        print("Training completed.")

    def transport(self, source: np.ndarray, reverse: bool = False) -> np.ndarray:
        """
        Transport source points using the trained transport map.

        Parameters:
        source (np.ndarray): Source points to transport.
        reverse (bool): Whether to use reverse transport map. Default is False.

        Returns:
        np.ndarray: Transported points.
        """
        source = torch.tensor(source, dtype=torch.float32, device=self.device).requires_grad_(True)
        if reverse:
            if self.positive:
                transported = F.relu(self.f.transport(source)).detach().cpu().numpy()
            else:
                transported = self.f.transport(source).detach().cpu().numpy()
        else:
            if self.positive:
                transported = F.relu(self.g.transport(source)).detach().cpu().numpy()
            else:
                transported = self.g.transport(source).detach().cpu().numpy()
        return transported
    
    def plot_2dpotential(self, resolution=100):
        import numpy as np
        import matplotlib.pyplot as plt

        # make the plot range is suitable with source data and target data
        x_range = (min(self.dataset.source[:, 0].min(), self.dataset.target[:, 0].min())-0.1,
                   max(self.dataset.source[:, 0].max(), self.dataset.target[:, 0].max())+0.1)
        y_range = (min(self.dataset.source[:, 1].min(), self.dataset.target[:, 1].min()-0.1),
                   max(self.dataset.source[:, 1].max(), self.dataset.target[:, 1].max())+0.1)

        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        grid_points = np.column_stack((X.ravel(), Y.ravel()))
        grid_points_tensor = torch.FloatTensor(grid_points).to(self.device)

        self.g.eval()
        with torch.no_grad():
            Z = self.g(grid_points_tensor).cpu().numpy().reshape(X.shape)

        plt.figure(figsize=(10, 8))
        contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
        plt.colorbar(contour, label='Potential')

        contour_lines = plt.contour(X, Y, Z, levels=20, colors='k', alpha=0.5, linewidths=0.5)
        plt.clabel(contour_lines, inline=True, fontsize=8)

        plt.title('2D Potential Map')
        plt.xlabel('X')
        plt.ylabel('Y')

        # Plot source and target points
        plt.scatter(self.dataset.source[:, 0], self.dataset.source[:, 1], c='b', s=10,
                    label='Source')
        plt.scatter(self.dataset.target[:, 0], self.dataset.target[:, 1], c='r', s=10,
                    label='Target')
        plt.legend(loc='upper right')

        plt.show()

    def _save_checkpoint(self, iteration, best_val_w2):
        checkpoint = {
            'iteration': iteration,
            'f_state_dict': self.f.state_dict(),
            'g_state_dict': self.g.state_dict(),
            'optimizer_f_state_dict': self.optimizer_f.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'best_val_w2': best_val_w2
        }
        torch.save(checkpoint, os.path.join(self.path, 'checkpoint.pth'))

    def _load_checkpoint(self):
        checkpoint_path = os.path.join(self.path, 'checkpoint.pth')
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        print("Checkpoint loaded.")
        return checkpoint
    
    def _save_model(self, path):
        self.network_config['f_state_dict'] = self.f.state_dict()
        self.network_config['g_state_dict'] = self.g.state_dict()
        torch.save(self.network_config, os.path.join(path, 'w2ot_networks.pt'))

    def load(self, path):
        self.network_config = torch.load(path)
        self.f = ICNN(**self.network_config['f'])
        self.g = ICNN(**self.network_config['g'])
        self.f.load_state_dict(self.network_config['f_state_dict'])
        self.g.load_state_dict(self.network_config['g_state_dict'])
        self.f = self.f.to(self.device)
        self.g = self.g.to(self.device)
    
    def save(self, path):
        self._save_model(path)
        
    # below are borrowed from cellot: https://github.com/bunnech/cellot/blob/main/cellot/models/cellot.py

    def _compute_loss_g(self, f, g, source, transport=None):
        if transport is None:
            transport = g.transport(source)

        return f(transport) - torch.multiply(source, transport).sum(-1, keepdim=True)


    def _compute_loss_f(self, f, g, source, target, transport=None):
        if transport is None:
            transport = g.transport(source)

        return  f(target) - f(transport)

    def compute_w2_distance(self, f, g, source, target, transport=None):
        if transport is None:
            transport = g.transport(source).squeeze()

        with torch.no_grad():
            Cpq = (source * source).sum(1, keepdim=True) + (target * target).sum(
                1, keepdim=True
            )
        Cpq = 0.5 * Cpq

        cost = (
            f(transport)
            - torch.multiply(source, transport).sum(-1, keepdim=True)
            - f(target)
            + Cpq
        )
        cost = cost.mean()
        return cost
