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

    def check_lipschitz(self, num_samples=1000, epsilon=1e-6):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            # Generate random input pairs
            x1 = torch.randn(num_samples, self.in_features)
            x2 = x1 + epsilon * torch.randn(num_samples, self.in_features)

            # Compute outputs
            y1 = self(x1)
            y2 = self(x2)

            # Compute Lipschitz constants
            input_distances = torch.norm(x1 - x2, dim=1)
            output_distances = torch.norm(y1 - y2, dim=1)
            lipschitz_constants = output_distances / input_distances

            max_lipschitz = lipschitz_constants.max().item()
            is_1_lipschitz = max_lipschitz <= self.scale + 1e-5  # Allow for small numerical errors

            print(f"Max Lipschitz constant: {max_lipschitz}")
            print(f"Is 1-Lipschitz: {is_1_lipschitz}")

            return is_1_lipschitz, max_lipschitz


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

    def check_lipschitz(self, num_samples=100000, epsilon=1e-6):
        self.eval()
        with torch.no_grad():
            # Check Lipschitz w.r.t x
            x1 = torch.randn(num_samples, self.in_features)
            x2 = x1 + epsilon * torch.randn(num_samples, self.in_features)
            y = torch.randn(num_samples, self.u_features)  # Same y for both x1 and x2

            y1_x = self(x1, y)[0]
            y2_x = self(x2, y)[0]

            input_distances_x = torch.norm(x1 - x2, dim=1)
            output_distances_x = torch.norm(y1_x - y2_x, dim=1)
            lipschitz_constants_x = output_distances_x / input_distances_x

            max_lipschitz_x = lipschitz_constants_x.max().item()
            is_1_lipschitz_x = max_lipschitz_x <= self.scale + 1e-5

            # Check Lipschitz w.r.t y
            x = torch.randn(num_samples, self.in_features)  # Same x for both y1 and y2
            y1 = torch.randn(num_samples, self.u_features)
            y2 = y1 + epsilon * torch.randn(num_samples, self.u_features)

            y1_y = self(x, y1)[1]
            y2_y = self(x, y2)[1]

            input_distances_y = torch.norm(y1 - y2, dim=1)
            output_distances_y = torch.norm(y1_y - y2_y, dim=1)
            lipschitz_constants_y = output_distances_y / input_distances_y

            max_lipschitz_y = lipschitz_constants_y.max().item()
            is_1_lipschitz_y = max_lipschitz_y <= self.scale + 1e-5

            print(f"Max Lipschitz constant w.r.t x: {max_lipschitz_x}")
            print(f"Is 1-Lipschitz w.r.t x: {is_1_lipschitz_x}")
            print(f"Max Lipschitz constant w.r.t y: {max_lipschitz_y}")
            print(f"Is 1-Lipschitz w.r.t y: {is_1_lipschitz_y}")

            return (is_1_lipschitz_x, max_lipschitz_x), (is_1_lipschitz_y, max_lipschitz_y)

