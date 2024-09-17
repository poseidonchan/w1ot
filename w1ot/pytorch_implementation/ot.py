import os
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.optim as optim
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


class OTDataset(torch.utils.data.Dataset):
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
    
    def sample_target(self, batch_size: int):
        idx = torch.randperm(self.target.size(0))[:batch_size]
        return self.target[idx]

    def sample_source(self, batch_size: int):
        idx = torch.randperm(self.source.size(0))[:batch_size]
        if self.transport_direction is not None:
            return self.source[idx], self.transport_direction[idx]
        else:
            return self.source[idx]
    
    def __len__(self):
        return self.source.size(0)
    
    def __getitem__(self, idx):
        return self.source[idx], self.target[idx]

class w1ot:
    def __init__(self,
                 source: np.ndarray = None,
                 target: np.ndarray = None,
                 device = None,
                 model_name = None):
        """
        Initialize the Wasserstein-1 optimal transport model.
        :param source: data sampled from the source distribution
        :param target: data sampled from the target distribution
        :param validation_size: validation size, between 0 and 1
        :param device: computing device of pytorch backend
        """
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_name is not None:
            self.path = os.path.join('./saved_models/w1ot', model_name)
        else:
            self.path = './saved_models/w1ot'
        
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if source is not None and target is not None:
            self.dataset = OTDataset(source, target, device=self.device)
            self.p = source.shape[1]
        else:
            self.dataset = None

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

    def fit_potential_function(self,
                           hidden_sizes: List[int] = [64]*4,
                           orthornormal_layer: str = 'cayley',
                           groups: int = 2,
                           batch_size: int = 256,
                           num_iters: int = 10000,
                           lr_init: float = 1e-3,
                           lr_min: float = 1e-5,
                           betas: Tuple[float, float] = (0.5, 0.5),
                           checkpoint_interval: int = 1000,
                           resume_from_checkpoint: bool = True) -> None:
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
                print(f"Resuming from iteration {start_iter}")
            except:
                print(f"Could not load checkpoint. Training from scratch.")

        if start_iter > 0:
            for _ in range(start_iter):
                scheduler.step()
        
        self.phi.train()
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
                self._save_checkpoint('potential', iteration + 1, self.phi, optimizer_phi, scheduler)
    

    def fit_distance_function(self,
                              d_hidden_sizes: List[int] = [64]*4,
                              eta_hidden_sizes: List[int] = [64]*4,
                              batch_size: int = 256,
                              num_iters: int = 10000,
                              lr_init: float = 1e-3,
                              lr_min: float = 1e-5,
                              betas: Tuple[float, float] = (0.5, 0.9),
                              checkpoint_interval: int = 1000,
                              resume_from_checkpoint: bool = True) -> None:
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

        progress_bar = tqdm(range(start_iter, num_iters), desc="Training")

        source = self.dataset.source.requires_grad_(True)
        (direction,) = torch.autograd.grad(
            self.phi(source),
            source,
            create_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((source.size()[0], 1), device=source.device).float(),
        )
        self.dataset.transport_direction = direction.detach()
        self.dataset.source.requires_grad_(False)

        for iteration in progress_bar:
            source_batch, direction_batch = self.dataset.sample_source(batch_size)
            target_batch = self.dataset.sample_target(batch_size)
            
            optimizer_eta.zero_grad()
            optimizer_D.zero_grad()

            # Transport the source batch using the current alpha network
            transported_batch = source_batch - self.eta(source_batch) * direction_batch

            # Discriminator loss: Use log loss
            discriminator_loss = - torch.mean(torch.log(self.D(target_batch) + 1e-6)) - torch.mean(
                torch.log(1 - self.D(transported_batch.detach()) + 1e-6))

            # Update the discriminator
            discriminator_loss.backward()
            optimizer_D.step()

            # Generator (Transport) loss: Use log loss to fool the discriminator
            D_fake = self.D(transported_batch)  # Recompute D_fake for the generator
            generator_loss = - torch.mean(torch.log(D_fake + 1e-6))

            # Update the alpha network
            generator_loss.backward()
            optimizer_eta.step()

            scheduler_D.step()
            scheduler_eta.step()

            progress_bar.set_postfix({
                'D Loss': f'{discriminator_loss.item():.4f}',
                'G Loss': f'{generator_loss.item():.4f}',
            })
                
            if (iteration + 1) % checkpoint_interval == 0:
                self._save_checkpoint('distance', iteration + 1, self.eta, optimizer_eta, scheduler_eta, 
                                      self.D, optimizer_D, scheduler_D)

        self._save_model(self.path)



    def transport(self,
                  source: np.ndarray,
                  method: str = 'GAN') -> np.ndarray:
        """
        Transport the source data to the target data using the trained transport map.
        :param source: data points in the source distribution
        :param method: "grad_guidance" or "neural_transport"
        :return: transported source data under wasserstein-1 optimal transport
        """
        if method == 'GAN':
            return self._transport_GAN(source)


    def _transport_GAN(self, source: np.ndarray) -> np.ndarray:
        """
        Transport the source data to the target data using the gradient guidance method.
        The transported sample is computed as x - alpha(x) * grad(phi(x)). where alpha(x) is the distance function
        :param source:
        :return:
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

        transported = source - self.eta(source) * direction

        return transported.cpu().detach().numpy()

    def _save_model(self, path):
        
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

    def plot_2dpotential(self, resolution=100):
        """
        Plot the 2D potential map.
        :param resolution: resolution of the plot
        :return: None
        """
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
        plt.scatter(self.dataset.source[:, 0], self.dataset.source[:, 1], c='r', s=10,
                    label='Source')
        plt.scatter(self.dataset.target[:, 0], self.dataset.target[:, 1], c='b', s=10,
                    label='Target')
        plt.legend(loc='upper right')

        plt.show()



class w2ot:
    def __init__(self,
                 source: np.ndarray = None,
                 target: np.ndarray = None,
                 device = None,
                 model_name = None):
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_name is not None:
            self.path = os.path.join('./saved_models/w2ot', model_name)
        else:
            self.path = './saved_models/w2ot'
        
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if source is not None and target is not None:
            self.dataset = OTDataset(source, target, device=self.device)
            self.p = source.shape[1]
        else:
            self.dataset = None
        
        self.f = None
        self.g = None

        self.network_config = {}
    
    def fit_potential_function(self,
                               hidden_sizes: List[int] = [64]*4,
                               batch_size: int = 256,
                               num_iters: int = 100000,
                               num_inner_iter: int = 10,
                               lr_init: float = 1e-3,
                               lr_min: float = 1e-4,
                               betas: Tuple[float, float] = (0.5, 0.9),
                               kernel_init: str = None,
                               resume_from_checkpoint: bool = True,
                               checkpoint_interval: int = 1000,
                               **kwargs):

        # Initialize the ICNN
        self.network_config['f'] = {'input_size': self.p, 'hidden_sizes': hidden_sizes, 'fnorm_penalty': 0, 'kernel_init': kernel_init, 'b': 0.1, 'std': 0.1}
        self.network_config['g'] = {'input_size': self.p, 'hidden_sizes': hidden_sizes, 'fnorm_penalty': 1, 'kernel_init': kernel_init, 'b': 0.1, 'std': 0.1}
        self.f = ICNN(**self.network_config['f']).to(self.device)
        self.g = ICNN(**self.network_config['g']).to(self.device)
        # Initialize the optimizers
        self.optimizer_f = optim.Adam(self.f.parameters(), lr=lr_init, betas=betas)
        self.optimizer_g = optim.Adam(self.g.parameters(), lr=lr_init, betas=betas)

        self.scheduler_f = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_f, T_max=num_iters, eta_min=lr_min)
        self.scheduler_g = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_g, T_max=num_iters, eta_min=lr_min)

        start_iter = 0
        if resume_from_checkpoint:
            try:
                checkpoint = self._load_checkpoint()
                self.f.load_state_dict(checkpoint['f_state_dict'])
                self.g.load_state_dict(checkpoint['g_state_dict'])
                self.optimizer_f.load_state_dict(checkpoint['optimizer_f_state_dict'])
                self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
                start_iter = checkpoint['iteration']
                print(f"Resuming from iteration {start_iter}")
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
            self.scheduler_f.step()
            self.scheduler_g.step()

            w2_distance = self.compute_w2_distance(self.f, self.g, source_batch, target_batch)

            progress_bar.set_postfix({
                'W2': f'{w2_distance.item():.2f}',
            })

            if (iteration + 1) % checkpoint_interval == 0:
                self._save_checkpoint(iteration + 1)
        self._save_model(self.path)
        print("Training completed.")

    def transport(self, source: np.ndarray, reverse: bool = False) -> np.ndarray:
        source = torch.tensor(source, dtype=torch.float32, device=self.device).requires_grad_(True)
        if reverse:
            transported = self.f.transport(source).detach().cpu().numpy()
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
        plt.scatter(self.dataset.source[:, 0], self.dataset.source[:, 1], c='r', s=10,
                    label='Source')
        plt.scatter(self.dataset.target[:, 0], self.dataset.target[:, 1], c='b', s=10,
                    label='Target')
        plt.legend(loc='upper right')

        plt.show()

    def _save_checkpoint(self, iteration):
        checkpoint = {
            'iteration': iteration,
            'f_state_dict': self.f.state_dict(),
            'g_state_dict': self.g.state_dict(),
            'optimizer_f_state_dict': self.optimizer_f.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
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
