import math
from typing import Sequence

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """Simple feed-forward autoencoder for fixed-size feature vectors.

    Args:
        input_dim: dimensionality of input feature vector
        hidden_dims: sequence of hidden layer sizes for encoder; decoder mirrors encoder
    """

    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (64, 32)):
        super().__init__()
        self.input_dim = input_dim
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(prev, h))
            enc_layers.append(nn.ReLU(inplace=True))
            prev = h
        self.encoder = nn.Sequential(*enc_layers)

        # decoder mirrors encoder
        dec_layers = []
        hidden_rev = list(hidden_dims)[::-1]
        for h in hidden_rev:
            dec_layers.append(nn.Linear(prev, h))
            dec_layers.append(nn.ReLU(inplace=True))
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon


def build_autoencoder_from_config(input_dim: int, hidden_dims_cfg: Sequence[int] | None = None):
    if hidden_dims_cfg is None:
        hidden_dims_cfg = (64, 32) if input_dim > 32 else (max(16, input_dim // 2),)
    return Autoencoder(input_dim, hidden_dims_cfg)
