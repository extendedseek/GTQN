from __future__ import annotations
import torch
import torch.nn as nn
from gtqn.models.modules import GraphConv, MLP

class CGGEncoder(nn.Module):
    """Collaborative Governance Graph encoder (training-only)."""
    def __init__(self, embed_dim: int, global_dim: int, dropout: float = 0.0):
        super().__init__()
        self.node_enc = MLP(embed_dim, 2*embed_dim, global_dim, n_layers=2, dropout=dropout)
        self.gconv = GraphConv(global_dim, global_dim)

    def forward(self, s: torch.Tensor, adj: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        h = self.node_enc(s)
        h = self.gconv(h, adj)
        m = active_mask.unsqueeze(-1)
        pooled = (h * m).sum(dim=1) / torch.clamp(m.sum(dim=1), min=1.0)
        return pooled
