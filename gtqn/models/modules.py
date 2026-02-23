from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        dims = [in_dim] + [hidden] * (n_layers - 1) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:, :T, :]

class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.enc(x, src_key_padding_mask=src_key_padding_mask)

def normalized_adj(adj: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if adj.dim() == 2:
        deg = torch.sum(adj, dim=-1)
        inv_sqrt = torch.rsqrt(deg + eps)
        D = torch.diag(inv_sqrt)
        return D @ adj @ D
    deg = torch.sum(adj, dim=-1)
    inv_sqrt = torch.rsqrt(deg + eps)
    D = torch.diag_embed(inv_sqrt)
    return D @ adj @ D

class GraphConv(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        A = adj + torch.eye(adj.size(-1), device=adj.device).unsqueeze(0)
        A = normalized_adj(A)
        return F.relu(self.lin(A @ x))
