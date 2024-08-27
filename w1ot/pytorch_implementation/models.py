import torch.nn as nn
import numpy as np
from .layers import LBlinear, LBlayer, PLBlayer
class LBNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, activation='relu', final_activation=None, scale=1.0):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes

        # Create layers
        layers = []
        prev_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                layers.append(LBlayer(prev_size, hidden_size, scale=np.sqrt(scale), activation=activation))
            else:
                layers.append(LBlayer(prev_size, hidden_size, scale=1, activation=activation))
            prev_size = hidden_size

        layers.append(LBlinear(prev_size, output_size, scale=np.sqrt(scale)))

        if final_activation == 'softplus':
            layers.append(nn.Softplus())


        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PLBNN(nn.Module):
    def __init__(self, x_size, y_size, output_size, hidden_sizes, activation='celu', scale=1.0):
        super().__init__()

        self.x_size = x_size
        self.y_size = y_size
        self.hidden_sizes = hidden_sizes

        layers = []
        prev_h_size = x_size
        u_size = y_size
        for h_size in hidden_sizes:
            layers.append(PLBlayer(prev_h_size, h_size, u_size, scale=scale, activation=activation))
            prev_h_size = h_size
            u_size = h_size

        self.layers = nn.ModuleList(layers)

        # Final linear layer
        self.final_layer = LBlinear(prev_h_size, output_size)

    def forward(self, x, y):
        h, u = x, y
        for layer in self.layers:
            h, u = layer(h, u)
        return self.final_layer(h)


class DNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, final_activation=None):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        if final_activation is None:
            layers.append(nn.Identity())
        elif final_activation=='softplus':
            layers.append(nn.Softplus())
        elif final_activation=='sigmoid':
            layers.append(nn.Sigmoid())
        else:
            raise ValueError("Final activation must be 'softplus', 'sigmoid', or None")

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x