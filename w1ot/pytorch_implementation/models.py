import torch
import torch.nn as nn
import numpy as np
from .layers import BjorckLayer, CayleyLayer

from typing import List

class LBNN(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 output_size: int = 1, 
                 hidden_sizes: List[int] = [64, 64, 64, 64], 
                 scale: float = 1.0, 
                 orthornormal_layer: str='bjorck', 
                 groups: int=2):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes

        if orthornormal_layer == 'bjorck':
            orthornormal_layer = BjorckLayer
        elif orthornormal_layer == 'cayley':
            orthornormal_layer = CayleyLayer
        else:
            raise ValueError("Orthornormal layer must be 'bjorck' or 'cayley'")

        # Create layers
        layers = []
        prev_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                layers.append(orthornormal_layer(prev_size, hidden_size, scale=np.sqrt(scale), groups=groups))
            else:
                layers.append(orthornormal_layer(prev_size, hidden_size, scale=1, groups=groups))
            prev_size = hidden_size

        layers.append(orthornormal_layer(prev_size, output_size, scale=np.sqrt(scale), groups=1))


        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        


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


# below are modified from cellot: https://github.com/bunnech/cellot/blob/main/cellot/networks/icnns.py

class ICNN(nn.Module):
    def __init__(
        self,
        input_size,
        output_size = 1,
        hidden_sizes = None,
        fnorm_penalty=0,
        kernel_init=None,
        b=0,
        std=0.1,
    ):

        super(ICNN, self).__init__()
        self.fnorm_penalty = fnorm_penalty
        
        self.sigma = nn.LeakyReLU(0.2)

        units = hidden_sizes + [1]

        self.W = nn.ModuleList(
            [
                nn.Linear(idim, odim, bias=False)
                for idim, odim in zip(units[:-1], units[1:])
            ]
        )

        self.A = nn.ModuleList(
            [nn.Linear(input_size, odim, bias=True) for odim in units]
        )

        if kernel_init is not None:

            for layer in self.A:
                if kernel_init == 'uniform':
                    nn.init.uniform_(layer.weight, b=b)
                elif kernel_init == 'normal':
                    nn.init.normal_(layer.weight, std=std)
                else:
                    pass
                nn.init.zeros_(layer.bias)

            for layer in self.W:
                if kernel_init == 'uniform':
                    nn.init.uniform_(layer.weight, b=b)
                elif kernel_init == 'normal':
                    nn.init.normal_(layer.weight, std=std)

    def forward(self, x):

        z = self.sigma(self.A[0](x))
        z = z * z

        for W, A in zip(self.W[:-1], self.A[1:-1]):
            z = self.sigma(W(z) + A(x))

        y = self.W[-1](z) + self.A[-1](x)

        return y

    def transport(self, x):
        assert x.requires_grad

        (output,) = torch.autograd.grad(
            self.forward(x),
            x,
            create_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((x.size()[0], 1), device=x.device).float(),
        )
        return output

    def clamp_w(self):
        for w in self.W:
            w.weight.data.clamp_(min=0)

    def penalize_w(self):
        return self.fnorm_penalty * sum(
            map(lambda x:  torch.nn.functional.relu(-x.weight).norm(), self.W)
        )