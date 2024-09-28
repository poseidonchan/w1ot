import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization
from torch import Tensor
from typing import Optional, Callable, Tuple

def process_group_size(x , num_units , axis = -1):
    # https://github.com/cemanil/LNets/tree/master/lnets/models/activations
    size = list(x.size()) # torch.tensor of shape B x C_out
    num_channels = size[axis] # C_out
    if num_channels % num_units != 0:
        raise ValueError("num channels is {}, but num units is {}".format(num_channels, num_units))
    
    size[axis] = -1
    if axis == -1:
        size += [num_channels//num_units]  
    else:
        size.insert(axis + 1, num_channels//num_units)
        
    return size

class GroupSort(torch.nn.Module):
    # https://github.com/cemanil/LNets/blob/master/lnets/models/activations/group_sort.py
    def __init__(self, num_units, axis=-1):
 
        super().__init__()
        self.num_units = num_units
        self.axis = axis
        
    def forward(self, x):
        assert x.shape[1] % self.num_units == 0
        size = process_group_size(x, self.num_units, self.axis) # torch.tensor of shape B x -1 x (C_out / num_units)
        grouped_x = x.view(*size)
        sort_dim = self.axis  if self.axis == -1 else axis + 1
        sorted_grouped_x, _ = grouped_x.sort(dim = sort_dim,descending = True) # torch.tensor of shape B x n_units x n_in_group
        sorted_x = sorted_grouped_x.view(*[x.shape[0],x.shape[1]])
        return sorted_x

def n_activation(x: torch.Tensor, theta: torch.Tensor):
    # x.shape e.g. [bs, c, h, w] or [bs, c].
    # theta.shape [c, 2].

    theta_sorted, _ = torch.sort(theta, dim=1)
    for _ in range(len(x.shape) - 2):
        theta_sorted = theta_sorted[..., None]

    line1 = x - 2 * theta_sorted[:, 0]
    line2 = -x
    line3 = x - 2 * theta_sorted[:, 1]

    piece1 = line1
    piece2 = torch.where(
        torch.less(x, theta_sorted[:, 0]),
        piece1,
        line2,
    )
    piece3 = torch.where(
        torch.less(x, theta_sorted[:, 1]),
        piece2,
        line3,
    )

    result = piece3
    return result

class BaseNActivation(nn.Module):
    def __init__(self,
                 in_channels: int,
                 initializer: Callable,
                 trainable: bool = True,
                 lr_factor: float = 1.,  # Changes grad/theta ratio.
                 ):
        super().__init__()

        self.sqrt_lr_factor = lr_factor ** 0.5

        theta_init_values = initializer(shape=(in_channels, 2), device=None)
        theta_init_values = theta_init_values / self.sqrt_lr_factor
        self.theta = nn.Parameter(theta_init_values, requires_grad=trainable)

    def forward(self, x: torch.Tensor):
        # x.shape e.g. [bs, c, h, w] or [bs, c].
        theta = self.theta * self.sqrt_lr_factor
        return n_activation(x, theta)

class RandomLogUniformNActivation(BaseNActivation):
    def __init__(self,
                 *args,
                 log_interval: Tuple[float, float] = (-5., 0.),  # log10
                 base: float = 10.,
                 **kwargs,
                 ):

        initializer = RandomThetaInitializer(log_interval, base)
        super().__init__(*args, initializer=initializer, **kwargs)

class RandomThetaInitializer:
    def __init__(self,
                 log_interval: Tuple[float, float] = (-5., 0.),
                 base: float = 10.
                 ):
        self.log_interval = log_interval
        self.base = base

    def __call__(self, shape, device):
        uniform01 = torch.rand(shape, device=device)
        l, h = self.log_interval
        log_theta_init = l + (h - l) * uniform01
        unsigned_theta_init = self.base ** log_theta_init
        signs = torch.tensor([-1., 1.], device=device)
        theta_values = unsigned_theta_init * signs
        return theta_values

class BjorckLayer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1, groups=1):
        super().__init__(in_features, out_features, bias)
        self.scale = scale
        
        self.activation = GroupSort(groups)

        torch.nn.init.uniform_(self.bias, -1, 1)
        torch.nn.init.kaiming_normal_(self.weight)
        
        
    def forward(self, x):
        scaling = torch.tensor([np.sqrt(self.weight.shape[0] * self.weight.shape[1])]).float().to(self.weight.device)
        if self.training:
            ortho_w = bjorck_orthonormalize(self.weight.t()/scaling,
                                            beta=0.5,
                                            iters=20,
                                            order=2).t()
        else:
            ortho_w = bjorck_orthonormalize(self.weight.t()/scaling,
                                            beta=0.5,
                                            iters=100,
                                            order=4).t()

        x = F.linear(self.scale*x, ortho_w, self.bias)
        
        x = self.activation(x)
        return x


class CayleyLayer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0, groups=1):
        super().__init__(in_features, out_features, bias)
        self.activation = GroupSort(groups)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm()
        self.scale = scale
        self.Q = None
        torch.nn.init.uniform_(self.bias, -1, 1)
            
    def forward(self, x):
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach()
        x = F.linear(self.scale * x, Q, self.bias)
        x = self.activation(x)
        return x

class SandwichLayer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0, groups=1):
        super().__init__(in_features+out_features, out_features, bias)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm() 
        self.scale = scale 
        self.psi = nn.Parameter(torch.zeros(out_features, dtype=torch.float32, requires_grad=True))   
        self.Q = None

    def forward(self, x):
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach()
        x = F.linear(self.scale * x, Q[:, fout:]) # B*h 
        if self.psi is not None:
            x = x * torch.exp(-self.psi) * (2 ** 0.5) # sqrt(2) \Psi^{-1} B * h
        if self.bias is not None:
            x += self.bias
        x = F.relu(x) * torch.exp(self.psi) # \Psi z
        x = 2 ** 0.5 * F.linear(x, Q[:, :fout].T) # sqrt(2) A^top \Psi z
        return x

def cayley(W):
    if len(W.shape) == 2:
        return cayley(W[None])[0]
    _, cout, cin = W.shape 
    if cin > cout:
        return cayley(W.transpose(1, 2)).transpose(1, 2)
    U, V = W[:, :cin], W[:, cin:]
    I = torch.eye(cin, dtype=W.dtype, device=W.device)[None, :, :]
    A = U - U.conj().transpose(1, 2) + V.conj().transpose(1, 2) @ V
    iIpA = torch.linalg.solve(I + A, I)
    return torch.cat((iIpA @ (I - A), -2 * V @ iIpA), axis=1)


def bjorck_orthonormalize(w, beta=0.5, iters=20, order=1):
    """
    Björck orthonormalization with optimized computation for higher orders using Horner's method.
    """
    eye = torch.eye(w.size(1), device=w.device, dtype=w.dtype)
    for _ in range(iters):
        w_t_w = w.t().mm(w)

        if order == 1:
            w = (1 + beta) * w - beta * w.mm(w_t_w)

        elif order == 2:
            if beta != 0.5:
                raise ValueError("Order >1 requires beta=0.5")
            coeffs = [15/8, -5/4, 3/8]
            # Horner's method
            P = coeffs[2] * w_t_w + coeffs[1] * eye
            P = P.mm(w_t_w) + coeffs[0] * eye
            w = w.mm(P)

        elif order == 3:
            if beta != 0.5:
                raise ValueError("Order >1 requires beta=0.5")
            coeffs = [35/16, -35/16, 21/16, -5/16]
            P = coeffs[3] * w_t_w + coeffs[2] * eye
            for c in coeffs[1::-1]:
                P = P.mm(w_t_w) + c * eye
            w = w.mm(P)

        elif order == 4:
            if beta != 0.5:
                raise ValueError("Order >1 requires beta=0.5")
            coeffs = [315/128, -105/32, 189/64, -45/32, 35/128]
            P = coeffs[4] * w_t_w + coeffs[3] * eye
            for c in coeffs[2::-1]:
                P = P.mm(w_t_w) + c * eye
            w = w.mm(P)

        else:
            raise ValueError("The requested order for orthonormalization is not supported.")

    return w


