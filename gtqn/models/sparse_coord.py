from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from gtqn.models.modules import MLP

@dataclass
class CoordOutput:
    context: torch.Tensor
    gate_mask: torch.Tensor
    attn_weights: torch.Tensor

class TwoStageSparseCoordination(nn.Module):
    """
    Implements the manuscript's two-stage mechanism:
      1) discrete peer gating (top-K by learned score) -> binary g^{(i,j)}
      2) soft relevance weighting over selected peers  -> omega^{(i,j)}
    """
    def __init__(self, d: int, K: int, dropout: float, use_gumbel_topk: bool = False):
        super().__init__()
        self.K = int(K)
        self.use_gumbel_topk = bool(use_gumbel_topk)
        self.gate_mlp = MLP(in_dim=3*d, hidden=2*d, out_dim=1, n_layers=2, dropout=dropout)
        self.att_mlp = MLP(in_dim=3*d, hidden=2*d, out_dim=1, n_layers=2, dropout=dropout)

    def _pair_features(self, s: torch.Tensor) -> torch.Tensor:
        B, N, D = s.shape
        si = s.unsqueeze(2).expand(B, N, N, D)
        sj = s.unsqueeze(1).expand(B, N, N, D)
        return torch.cat([si, sj, si * sj], dim=-1)

    def forward(
        self,
        s: torch.Tensor,
        adj_allowed: Optional[torch.Tensor] = None,
        active_mask: Optional[torch.Tensor] = None,
        act_mask: Optional[torch.Tensor] = None,
    ) -> CoordOutput:
        B, N, D = s.shape
        pair = self._pair_features(s)
        scores = self.gate_mlp(pair).squeeze(-1)

        if adj_allowed is not None:
            scores = scores.masked_fill(adj_allowed <= 0, float("-inf"))
        if active_mask is not None:
            col_mask = (active_mask <= 0).unsqueeze(1).expand(B, N, N)
            scores = scores.masked_fill(col_mask, float("-inf"))
        if act_mask is not None:
            row_mask = (act_mask <= 0).unsqueeze(2).expand(B, N, N)
            scores = scores.masked_fill(row_mask, float("-inf"))

        K = min(self.K, N)
        topk_vals, topk_idx = torch.topk(scores, k=K, dim=-1)
        gate = torch.zeros((B, N, N), device=s.device, dtype=torch.float32)
        gate.scatter_(dim=-1, index=topk_idx, src=torch.ones_like(topk_vals, dtype=torch.float32))

        att_logits = self.att_mlp(pair).squeeze(-1)
        att_logits = att_logits.masked_fill(gate <= 0, float("-inf"))
        att_weights = torch.softmax(att_logits, dim=-1)
        att_weights = torch.nan_to_num(att_weights, nan=0.0)

        ctx = torch.einsum("bij,bij,bjd->bid", gate, att_weights, s)
        return CoordOutput(context=ctx, gate_mask=gate, attn_weights=att_weights)

class DenseCoordination(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.att_mlp = MLP(in_dim=3*d, hidden=2*d, out_dim=1, n_layers=2, dropout=dropout)

    def forward(self, s: torch.Tensor, adj_allowed=None, active_mask=None, act_mask=None) -> CoordOutput:
        B, N, D = s.shape
        si = s.unsqueeze(2).expand(B, N, N, D)
        sj = s.unsqueeze(1).expand(B, N, N, D)
        pair = torch.cat([si, sj, si * sj], dim=-1)
        logits = self.att_mlp(pair).squeeze(-1)

        if adj_allowed is not None:
            logits = logits.masked_fill(adj_allowed <= 0, float("-inf"))
        if active_mask is not None:
            col_mask = (active_mask <= 0).unsqueeze(1).expand(B, N, N)
            logits = logits.masked_fill(col_mask, float("-inf"))
        if act_mask is not None:
            row_mask = (act_mask <= 0).unsqueeze(2).expand(B, N, N)
            logits = logits.masked_fill(row_mask, float("-inf"))

        w = torch.softmax(logits, dim=-1)
        w = torch.nan_to_num(w, nan=0.0)
        ctx = torch.einsum("bij,bjd->bid", w, s)
        gate = torch.ones((B, N, N), device=s.device, dtype=torch.float32)
        return CoordOutput(context=ctx, gate_mask=gate, attn_weights=w)

class TopKOnlyCoordination(nn.Module):
    def __init__(self, d: int, K: int, dropout: float):
        super().__init__()
        self.K = int(K)
        self.gate_mlp = MLP(in_dim=3*d, hidden=2*d, out_dim=1, n_layers=2, dropout=dropout)

    def forward(self, s: torch.Tensor, adj_allowed=None, active_mask=None, act_mask=None) -> CoordOutput:
        B, N, D = s.shape
        si = s.unsqueeze(2).expand(B, N, N, D)
        sj = s.unsqueeze(1).expand(B, N, N, D)
        pair = torch.cat([si, sj, si * sj], dim=-1)
        scores = self.gate_mlp(pair).squeeze(-1)

        if adj_allowed is not None:
            scores = scores.masked_fill(adj_allowed <= 0, float("-inf"))
        if active_mask is not None:
            col_mask = (active_mask <= 0).unsqueeze(1).expand(B, N, N)
            scores = scores.masked_fill(col_mask, float("-inf"))
        if act_mask is not None:
            row_mask = (act_mask <= 0).unsqueeze(2).expand(B, N, N)
            scores = scores.masked_fill(row_mask, float("-inf"))

        K = min(self.K, N)
        _, idx = torch.topk(scores, k=K, dim=-1)
        gate = torch.zeros((B, N, N), device=s.device, dtype=torch.float32)
        gate.scatter_(dim=-1, index=idx, src=torch.ones_like(idx, dtype=torch.float32))
        w = gate / torch.clamp(gate.sum(dim=-1, keepdim=True), min=1.0)
        ctx = torch.einsum("bij,bjd->bid", w, s)
        return CoordOutput(context=ctx, gate_mask=gate, attn_weights=w)

class SoftOnlyCoordination(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.att_mlp = MLP(in_dim=3*d, hidden=2*d, out_dim=1, n_layers=2, dropout=dropout)

    def forward(self, s: torch.Tensor, adj_allowed=None, active_mask=None, act_mask=None) -> CoordOutput:
        B, N, D = s.shape
        si = s.unsqueeze(2).expand(B, N, N, D)
        sj = s.unsqueeze(1).expand(B, N, N, D)
        pair = torch.cat([si, sj, si * sj], dim=-1)
        logits = self.att_mlp(pair).squeeze(-1)

        if adj_allowed is not None:
            logits = logits.masked_fill(adj_allowed <= 0, float("-inf"))
        if active_mask is not None:
            col_mask = (active_mask <= 0).unsqueeze(1).expand(B, N, N)
            logits = logits.masked_fill(col_mask, float("-inf"))
        if act_mask is not None:
            row_mask = (act_mask <= 0).unsqueeze(2).expand(B, N, N)
            logits = logits.masked_fill(row_mask, float("-inf"))

        w = torch.softmax(logits, dim=-1)
        w = torch.nan_to_num(w, nan=0.0)
        ctx = torch.einsum("bij,bjd->bid", w, s)
        gate = (w > 0).float()
        return CoordOutput(context=ctx, gate_mask=gate, attn_weights=w)

class RandomCoordination(nn.Module):
    def __init__(self, d: int, K: int):
        super().__init__()
        self.K = int(K)

    def forward(self, s: torch.Tensor, adj_allowed=None, active_mask=None, act_mask=None) -> CoordOutput:
        B, N, D = s.shape
        gate = torch.zeros((B, N, N), device=s.device, dtype=torch.float32)
        for b in range(B):
            for i in range(N):
                if act_mask is not None and act_mask[b, i] <= 0:
                    continue
                candidates = torch.arange(N, device=s.device)
                if active_mask is not None:
                    candidates = candidates[active_mask[b] > 0]
                if adj_allowed is not None:
                    candidates = candidates[adj_allowed[b, i, candidates] > 0]
                if candidates.numel() == 0:
                    continue
                perm = candidates[torch.randperm(candidates.numel(), device=s.device)]
                sel = perm[: min(self.K, perm.numel())]
                gate[b, i, sel] = 1.0
        w = gate / torch.clamp(gate.sum(dim=-1, keepdim=True), min=1.0)
        ctx = torch.einsum("bij,bjd->bid", w, s)
        return CoordOutput(context=ctx, gate_mask=gate, attn_weights=w)
