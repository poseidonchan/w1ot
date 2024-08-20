import torch
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from typing import Dict, Any

from .models import LBNN, DNN, PLBNN


class w1ot:
    def __init__(self,
                 source: np.ndarray,
                 target: np.ndarray,
                 validation_size: float = 0.1,
                 device = None):
        """
        Initialize the Wasserstein-1 optimal transport model.
        :param source: data sampled from the source distribution
        :param target: data sampled from the target distribution
        :param validation_size: validation size, between 0 and 1
        :param device: computing device of pytorch backend
        """
        self.source = torch.tensor(source, dtype=torch.float32, device=device)
        self.target = torch.tensor(target, dtype=torch.float32, device=device)
        self.p = source.shape[1]

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.validation_size = None
        # Setup validation set
        if validation_size < 0 or validation_size > 1:
            raise ValueError("Validation size must be between 0 and 1.")
        else:
            self.validation_size = validation_size

        if self.validation_size == 0:
            self.source_train = self.source
            self.source_val = self.source
            self.target_train = self.target
            self.target_val = self.target
        else:
            n_source = source.shape[0]
            n_target = target.shape[0]
            source_val_size = int(0.1 * n_source)
            target_val_size = int(0.1 * n_target)
            self.source_train = self.source[:-source_val_size]
            self.source_val = self.source[-source_val_size:]
            self.target_train = self.target[:-target_val_size]
            self.target_val = self.target[-target_val_size:]

        # Kantorovich potential
        self.phi = None
        self.grad_phi_source = None

        # step size function
        self.alpha = None

        # discriminator
        self.D = None

    def shuffle_data(self,
                     data: torch.Tensor,
                     shuffle_indices = None):
        if shuffle_indices is None:
            shuffle_indices = torch.randperm(len(data))
            return  data[shuffle_indices], shuffle_indices
        else:
            return data[shuffle_indices]

    def maximize_potential(self,
                           phi_network_opt: Dict[str, Any] =  {"hidden_sizes": [32, 32, 32, 32],
                                                               "activation": "relu",
                                                               "scale": 1.0},
                           batch_size: int = None,
                           num_epochs: int = 200,
                           lr_init: float = 1e-2,
                           lr_min: float = 1e-4,
                           optimizer: str = 'adam') -> None:

        self.phi = LBNN(input_size=self.p, output_size=1, **phi_network_opt).to(self.device)
        print("the optimizer is: ", optimizer)
        if optimizer.lower() == 'lbfgs':
            self.maximize_potential_lbfgs(batch_size, num_epochs, lr_init, lr_min)
        elif optimizer.lower() == 'sgd' or optimizer.lower() == 'adam' or optimizer.lower() == 'rmsprop':
            self.maximize_potential_sgd(batch_size, num_epochs, lr_init, lr_min, optimizer)

        else:
            raise ValueError("This optimization method is not supported yet.")

    def maximize_potential_sgd(self,
                               batch_size: int,
                               num_epochs: int,
                               lr: float,
                               lr_min: float = 5e-5,
                               optimizer: str = None) -> None:
        """
        Maximize the Kantorovich potential.

        Args:
            batch_size (int): Minibatch size
            num_epochs (int): Number of training epochs
        """
        # Optimizer
        if optimizer.lower() == 'adam':
            optimizer_phi = optim.Adam(self.phi.parameters(), betas=(0.5, 0.9), lr=lr)
        elif optimizer.lower() == 'rmsprop':
            optimizer_phi = optim.RMSprop(self.phi.parameters(), lr=lr)
        elif optimizer.lower() == 'sgd':
            optimizer_phi = optim.SGD(self.phi.parameters(), lr=lr)
        else:
            raise ValueError("Optimizer must be 'sgd', 'adam', or 'rmsprop'.")

        # Initialize the cosine annealing scheduler
        scheduler = CosineAnnealingLR(optimizer_phi, T_max=num_epochs, eta_min=lr_min)

        for epoch in range(num_epochs):
            self.phi.train()
            train_loss = 0
            num_batches = 0

            self.source_train, _ = self.shuffle_data(self.source_train)
            self.target_train, _ = self.shuffle_data(self.target_train)

            for i in range(0, len(self.source_train), batch_size):

                source_batch = self.source_train[i:i + batch_size]
                target_batch = self.target_train[i:i + batch_size]
                # Check if the batch sizes are consistent, if not skip the last batch
                if len(source_batch) != len(target_batch):
                    break

                optimizer_phi.zero_grad()
                loss = -((self.phi(source_batch).mean() - self.phi(target_batch).mean()))
                loss.backward()
                optimizer_phi.step()
                train_loss += loss.item()
                num_batches += 1

            train_loss /= num_batches

            # Validation
            if self.validation_size == 0:
                val_loss = train_loss
            else:
                self.phi.eval()
                with torch.no_grad():
                    val_loss = -((self.phi(self.source_val).mean() - self.phi(self.target_val).mean()))

            # Step the scheduler
            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

    def maximize_potential_lbfgs(self, batch_size: int, num_epochs: int, lr: float, lr_min: float = 1e-3) -> None:
        """
        Maximize the Kantorovich potential using the LBFGS optimizer.

        Args:
            batch_size (int): Minibatch size. Note: with LBFGS, it's common to use the entire dataset or a large batch.
            num_epochs (int): Number of training epochs.
            lr (float): Learning rate.
        """
        # for param in self.phi.parameters():
        #     param.requires_grad = True

        # Define the LBFGS optimizer
        optimizer_phi = optim.LBFGS(self.phi.parameters(), lr=lr, max_iter=20, line_search_fn="strong_wolfe")
        # Initialize the cosine annealing scheduler
        scheduler = CosineAnnealingLR(optimizer_phi, T_max=num_epochs, eta_min=lr_min)

        def closure():
            optimizer_phi.zero_grad()
            # Using the entire dataset or a large batch
            source_batch = self.source_train
            target_batch = self.target_train
            loss = -((self.phi(source_batch).mean() - self.phi(target_batch).mean()))
            loss.backward()
            return loss

        for epoch in range(num_epochs):
            self.phi.train()

            self.source_train, _ = self.shuffle_data(self.source_train)
            self.target_train, _ = self.shuffle_data(self.target_train)

            # Perform optimization step
            optimizer_phi.step(closure)

            # Validation
            self.phi.eval()
            with torch.no_grad():
                val_loss = -((self.phi(self.source_val).mean() - self.phi(self.target_val).mean()))

            # Step the scheduler
            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch}: Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

    def fit_distance_function(self,
                              d_network_opt: Dict[str, Any] = {'hidden_sizes': [32, 32, 32, 32],
                                                               'final_activation': 'sigmoid'},
                              alpha_network_opt: Dict[str, Any] = {'hidden_sizes': [32, 32, 32, 32],
                                                                   'final_activation': 'softplus'},
                              batch_size: int = 64,
                              num_epochs: int = 1000,
                              lr_init: float = 1e-4,
                              lr_min: float = 1e-5,
                              optimizer: str = 'rmsprop') -> None:

        self.alpha = DNN(input_size=self.p, output_size=1, **alpha_network_opt).to(self.device)
        self.D = DNN(input_size=self.p, output_size=1, **d_network_opt).to(self.device)

        # Pre-compute gradient of phi in batches
        self.phi.eval()
        self.grad_phi_source = []
        with torch.enable_grad():
            for i in range(0, len(self.source), batch_size):
                source_batch = self.source[i:i + batch_size].clone().detach().requires_grad_(True)
                phi_output = self.phi(source_batch)
                grad_phi = torch.autograd.grad(phi_output.sum(), source_batch, create_graph=False)[0]
                self.grad_phi_source.append(grad_phi.detach())

        self.grad_phi_source = torch.cat(self.grad_phi_source)
        # Optimizers for alpha and D
        if optimizer == 'adam':
            optimizer_D = optim.Adam(self.D.parameters(), lr=lr_init)
            optimizer_alpha = optim.Adam(self.alpha.parameters(), lr=1e-4)
        elif optimizer == 'rmsprop':
            optimizer_D = optim.RMSprop(self.D.parameters(), lr=lr_init)
            optimizer_alpha = optim.RMSprop(self.alpha.parameters(), lr=1e-4)
        else:
            raise ValueError("Optimizer must be 'adam' or 'rmsprop'")

        scheduler_D = CosineAnnealingLR(optimizer_D, T_max=num_epochs, eta_min=lr_min)
        scheduler_alpha = CosineAnnealingLR(optimizer_alpha, T_max=num_epochs, eta_min=lr_min)

        for epoch in range(num_epochs):
            self.alpha.train()
            self.D.train()

            total_train_loss = 0
            total_discriminator_loss = 0
            total_generator_loss = 0
            num_batches = 0

            self.source, shuffle_indices = self.shuffle_data(self.source)
            self.target, _ = self.shuffle_data(self.target)
            self.grad_phi_source = self.shuffle_data(self.grad_phi_source, shuffle_indices)

            for i in range(0, len(self.source), batch_size):
                source_batch = self.source[i:i + batch_size]
                target_batch = self.target[i:i + batch_size]
                phi_grad_batch = self.grad_phi_source[i:i + batch_size]
                # Check if the batch sizes are consistent, if not skip the last batch
                if len(source_batch) != len(target_batch):
                    break

                optimizer_alpha.zero_grad()
                optimizer_D.zero_grad()

                # Re-attach phi gradient to the computational graph
                phi_grad_batch = phi_grad_batch.detach().requires_grad_()

                # Transport the source batch using the current alpha network
                transported_batch = source_batch - self.alpha(source_batch) * phi_grad_batch

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
                optimizer_alpha.step()

                total_discriminator_loss += discriminator_loss.item()
                total_generator_loss += generator_loss.item()
                total_train_loss += generator_loss.item()
                num_batches += 1
            avg_train_loss = total_train_loss / num_batches
            avg_discriminator_loss = total_discriminator_loss / num_batches
            avg_generator_loss = total_generator_loss / num_batches

            scheduler_D.step()
            scheduler_alpha.step()

            print(f"Epoch {epoch}: "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Discriminator Loss: {avg_discriminator_loss:.4f}, "
                  f"Generator Loss: {avg_generator_loss:.4f}"
                  f"LR_D: {scheduler_D.get_last_lr()[0]:.6f}",
                  f"LR_G: {scheduler_alpha.get_last_lr()[0]:.6}")

        print("Training of transport map completed.")



    def transport(self,
                  source: np.ndarray,
                  method: str = 'grad_guidance') -> np.ndarray:
        """
        Transport the source data to the target data using the trained transport map.
        :param source: data points in the source distribution
        :param method: "grad_guidance" or "neural_transport"
        :return: transported source data under wasserstein-1 optimal transport
        """
        if method == 'grad_guidance':
            return self.transport_grad_guidance(source)


    def transport_grad_guidance(self, source: np.ndarray) -> np.ndarray:
        """
        Transport the source data to the target data using the gradient guidance method.
        The transported sample is computed as x - alpha(x) * grad(phi(x)). where alpha(x) is the distance function
        :param source:
        :return:
        """
        self.alpha.eval()
        self.phi.eval()
        transported = []

        source = torch.tensor(source, dtype=torch.float32, device=self.device)

        for sample in source:
            sample = sample.unsqueeze(0).requires_grad_(True)
            with torch.enable_grad():
                # Compute the gradient of phi
                grad_phi = torch.autograd.grad(self.phi(sample), sample)[0]
                # Use the trained alpha network to compute the scalar for the gradient
                alpha_sample = self.alpha(sample)
                # Compute the transported sample
                transported_sample = sample - alpha_sample * grad_phi
                transported.append(transported_sample.detach())

        return torch.cat(transported).cpu().detach().numpy()

    def plot_2dpotential(self, resolution=100):
        """
        Plot the 2D potential map.
        :param resolution: resolution of the plot
        :return: None
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # make the plot range is suitable with source data and target data
        x_range = (min(self.source[:, 0].min().item(), self.target[:, 0].min().item())-1,
                   max(self.source[:, 0].max().item(), self.target[:, 0].max().item())+1)
        y_range = (min(self.source[:, 1].min().item(), self.target[:, 1].min().item()-1),
                   max(self.source[:, 1].max().item(), self.target[:, 1].max().item())+1)

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
        plt.scatter(self.source[:, 0].detach().cpu().numpy(), self.source[:, 1].detach().cpu().numpy(), c='r', s=10,
                    label='Source')
        plt.scatter(self.target[:, 0].detach().cpu().numpy(), self.target[:, 1].detach().cpu().numpy(), c='b', s=10,
                    label='Target')
        plt.legend(loc='upper right')

        plt.show()