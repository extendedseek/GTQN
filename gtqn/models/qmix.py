from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class QMixMixer(nn.Module):
    """QMIX-style monotonic mixing network conditioned on a global embedding."""
    def __init__(self, n_agents: int, state_dim: int, embed_dim: int = 64):
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        self.hyper_w1 = nn.Sequential(nn.Linear(state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, n_agents * embed_dim))
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_w2 = nn.Sequential(nn.Linear(state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))

    def forward(self, q_agents: torch.Tensor, state_embed: torch.Tensor) -> torch.Tensor:
        B, N = q_agents.shape
        assert N == self.n_agents
        w1 = torch.abs(self.hyper_w1(state_embed)).view(B, N, self.embed_dim)
        b1 = self.hyper_b1(state_embed).view(B, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(q_agents.view(B, 1, N), w1) + b1)  # [B,1,E]
        w2 = torch.abs(self.hyper_w2(state_embed)).view(B, self.embed_dim, 1)
        b2 = self.hyper_b2(state_embed).view(B, 1, 1)
        y = torch.bmm(hidden, w2) + b2
        return y.view(B)
