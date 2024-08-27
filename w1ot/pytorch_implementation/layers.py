import torch
import torch.nn as nn
import torch.nn.functional as F

def cayley(W):
    if len(W.shape) == 2:
        return cayley(W[None])[0]
    _, cout, cin = W.shape
    if cin > cout:
        return cayley(W.transpose(1, 2)).transpose(1, 2)
    U, V = W[:, :cin], W[:, cin:]
    I = torch.eye(cin, dtype=W.dtype, device=W.device)[None, :, :]
    A = U - U.conj().transpose(1, 2) + V.conj().transpose(1, 2) @ V
    iIpA = torch.inverse(I + A)
    return torch.cat((iIpA @ (I - A), -2 * V @ iIpA), axis=1)


class LBlinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0, AB=False):
        super().__init__(in_features + out_features, out_features, bias)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm()
        self.scale = scale
        self.AB = AB
        self.Q = None

    def forward(self, x):
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach()
        x = F.linear(self.scale * x, Q[:, fout:])  # B @ x
        if self.AB:
            x = 2 * F.linear(x, Q[:, :fout].T)  # 2 A.T @ B @ x
        if self.bias is not None:
            x += self.bias
        return x


class LBlayer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0, activation='relu'):
        super().__init__(in_features + out_features, out_features, bias)
        self.in_features = in_features
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm()
        self.scale = scale
        self.psi = nn.Parameter(torch.zeros(out_features, dtype=torch.float32, requires_grad=True))
        self.Q = None

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'celu':
            self.activation = nn.CELU()
        elif activation == 'id':
            self.activation = nn.Identity()
        else:
            raise ValueError("Activation must be 'relu', 'celu', or 'id'")



    def forward(self, x):
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach()
        x = F.linear(self.scale * x, Q[:, fout:])  # B*h
        if self.psi is not None:
            x = x * torch.exp(-self.psi) * (2 ** 0.5)  # sqrt(2) \Psi^{-1} B * h
        if self.bias is not None:
            x += self.bias
        x = self.activation(x) * torch.exp(self.psi)  # \Psi z
        x = 2 ** 0.5 * F.linear(x, Q[:, :fout].T)  # sqrt(2) A^top \Psi z
        return x


class PLBlayer(nn.Module):
    def __init__(self, in_features, u_features, out_features, bias=True, scale=1.0, activation='relu'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.u_features = u_features
        self.scale = scale

        # Parameters for h update
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features + out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.psi = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

        # Parameters for u update
        self.W_u_in = nn.Parameter(torch.Tensor(in_features, u_features))
        self.bias_u_in = nn.Parameter(torch.Tensor(in_features))
        self.W_u_out = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_u_out = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()
        self.Q = None

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'celu':
            self.activation = nn.CELU()
        elif activation == 'id':
            self.activation = nn.Identity()
        else:
            raise ValueError("Activation must be 'relu', 'celu', or 'id'")

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        nn.init.kaiming_uniform_(self.W_u_in, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.W_u_out, a=5 ** 0.5)
        nn.init.zeros_(self.bias_u_in)
        nn.init.zeros_(self.bias_u_out)

    def forward(self, h, u):
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach()

        # Update u to h dimension
        u = F.linear(u, self.W_u_in) + self.bias_u_in

        # Update h
        h = F.linear(self.scale * (h + u), Q[:, fout:])  # B(h + u)
        if self.psi is not None:
            h = h * torch.exp(-self.psi) * (2 ** 0.5)  # sqrt(2) \Psi^{-1} B(h + u)
        if self.bias is not None:
            h += self.bias
        h = self.activation(h) * torch.exp(self.psi)  # \Psi z

        # Update u to next layer dimension
        u = F.linear(u, self.W_u_out) + self.bias_u_out

        h = 2 ** 0.5 * F.linear(h, Q[:, :fout].T) + u  # sqrt(2) A^top \Psi z + u

        return h, u


