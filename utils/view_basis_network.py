"""
Low-Rank View-Dependent Color Field (LR-VDCF)
Global shared view basis network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewBasisNetwork(nn.Module):
    """
    Global shared view basis network.
    Input: normalized view direction.
    Output: RGB values for K basis functions.
    """

    def __init__(self, K=6, hidden_dim=128, use_positional_encoding=True):
        super().__init__()
        self.K = K
        self.use_positional_encoding = use_positional_encoding

        if use_positional_encoding:
            self.pos_encoding_freq = 6
            self.pos_encoding_dim = 3 * 2 * self.pos_encoding_freq
            input_dim = self.pos_encoding_dim
        else:
            input_dim = 3

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, K * 3),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def positional_encoding(self, x, L=None):
        if L is None:
            L = self.pos_encoding_freq if hasattr(self, "pos_encoding_freq") else 6
        encoding = []
        for l in range(L):
            encoding.append(torch.sin(2 ** l * torch.pi * x))
            encoding.append(torch.cos(2 ** l * torch.pi * x))
        return torch.cat(encoding, dim=-1)

    def forward(self, view_dirs):
        view_dirs = F.normalize(view_dirs, dim=-1)
        encoded = self.positional_encoding(view_dirs) if self.use_positional_encoding else view_dirs
        output = self.network(encoded)
        basis = output.view(-1, self.K, 3)
        basis = torch.tanh(basis) * 0.6
        return basis


class LRVDCFColorModel:
    """
    Low-Rank View-Dependent Color Field color model.
    """

    def __init__(self, basis_network, K=6):
        self.basis_network = basis_network
        self.K = K

    def compute_color(self, base_colors, alpha_coeffs, view_dirs):
        basis = self.basis_network(view_dirs)
        n_points = base_colors.shape[0]
        basis = basis.reshape(n_points, -1)
        if basis.shape[1] % 3 != 0:
            pad = 3 - (basis.shape[1] % 3)
            basis = torch.cat(
                [basis, torch.zeros((n_points, pad), device=basis.device, dtype=basis.dtype)], dim=1
            )
        basis = basis.view(n_points, -1, 3)

        alpha_coeffs = alpha_coeffs.reshape(n_points, -1, 1)
        k = min(alpha_coeffs.shape[1], basis.shape[1])

        if k == 0:
            view_dependent = torch.zeros((n_points, 3), device=base_colors.device, dtype=base_colors.dtype)
        else:
            view_dependent = torch.sum(alpha_coeffs[:, :k] * basis[:, :k], dim=1)

        colors = base_colors + view_dependent
        colors = torch.clamp(colors, 0.0, 1.0)
        return colors


def create_basis_network(K=6, hidden_dim=128, device="cuda"):
    network = ViewBasisNetwork(K=K, hidden_dim=hidden_dim)
    return network.to(device)
