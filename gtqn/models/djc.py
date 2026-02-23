from __future__ import annotations
from typing import Optional, Dict
import torch
import torch.nn as nn

from gtqn.models.modules import MLP, SinusoidalPositionalEncoding, TransformerEncoder
from gtqn.models.sparse_coord import (
    TwoStageSparseCoordination, DenseCoordination, TopKOnlyCoordination, SoftOnlyCoordination, RandomCoordination
)

class DJC(nn.Module):
    """Distributed Junction Controller."""
    def __init__(
        self,
        obs_dim: int,
        history_len: int,
        embed_dim: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        n_actions: int,
        coord_variant: str,
        K: int,
        use_gumbel_topk: bool = False,
    ):
        super().__init__()
        self.embed = nn.Linear(obs_dim, embed_dim)
        self.pos = SinusoidalPositionalEncoding(embed_dim, max_len=max(64, history_len + 4))
        self.temporal = TransformerEncoder(embed_dim, n_heads, n_layers, dropout)

        v = coord_variant.lower()
        if v == "dense":
            self.coord = DenseCoordination(embed_dim, dropout)
        elif v == "topk_only":
            self.coord = TopKOnlyCoordination(embed_dim, K, dropout)
        elif v == "soft_only":
            self.coord = SoftOnlyCoordination(embed_dim, dropout)
        elif v == "random":
            self.coord = RandomCoordination(embed_dim, K)
        else:
            self.coord = TwoStageSparseCoordination(embed_dim, K, dropout, use_gumbel_topk=use_gumbel_topk)

        self.q_head = MLP(in_dim=2*embed_dim, hidden=2*embed_dim, out_dim=n_actions, n_layers=2, dropout=dropout)

    def forward(
        self,
        obs_hist: torch.Tensor,                 # [B,N,H,obs_dim]
        adj_allowed: Optional[torch.Tensor] = None,
        active_mask: Optional[torch.Tensor] = None,
        act_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, N, H, _ = obs_hist.shape
        x = self.embed(obs_hist).view(B*N, H, -1)
        x = self.temporal(self.pos(x))
        s = x[:, -1, :].view(B, N, -1)
        coord_out = self.coord(s, adj_allowed=adj_allowed, active_mask=active_mask, act_mask=act_mask)
        c = coord_out.context
        q = self.q_head(torch.cat([s, c], dim=-1))
        return {"s": s, "c": c, "q": q, "gate": coord_out.gate_mask, "attn": coord_out.attn_weights}
