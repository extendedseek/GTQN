from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn

from gtqn.models.djc import DJC
from gtqn.models.cgg import CGGEncoder
from gtqn.models.qmix import QMixMixer

class GTQNSystem(nn.Module):
    """DJC + CGG + Mixer."""
    def __init__(
        self,
        obs_dim: int,
        history_len: int,
        embed_dim: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        n_actions: int,
        N_max: int,
        coord_variant: str,
        K: int,
        use_gumbel_topk: bool,
        cgg_enabled: bool,
        mixer: str,
        hypernet_embed: int,
    ):
        super().__init__()
        self.N_max = N_max
        self.mixer_type = mixer
        self.djc = DJC(obs_dim, history_len, embed_dim, n_heads, n_layers, dropout, n_actions, coord_variant, K, use_gumbel_topk)
        self.cgg = CGGEncoder(embed_dim=embed_dim, global_dim=embed_dim, dropout=dropout)
        self.mixer = QMixMixer(n_agents=N_max, state_dim=embed_dim, embed_dim=hypernet_embed) if mixer == "qmix" else None

    def mix_q(self, q_agents: torch.Tensor, global_embed: torch.Tensor, active_mask: torch.Tensor, act_mask: torch.Tensor) -> torch.Tensor:
        q = q_agents * active_mask * act_mask
        if self.mixer_type == "qmix" and self.mixer is not None:
            return self.mixer(q, global_embed)
        return q.sum(dim=-1)

    def forward_batch(self, obs_hist: torch.Tensor, adj: torch.Tensor, active_mask: torch.Tensor, act_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        djc_out = self.djc(obs_hist, adj_allowed=adj, active_mask=active_mask, act_mask=act_mask)
        global_embed = self.cgg(djc_out["s"], adj, active_mask)
        greedy = torch.max(djc_out["q"], dim=-1).values
        q_tot = self.mix_q(greedy, global_embed, active_mask, act_mask)
        return {**djc_out, "global_embed": global_embed, "q_tot": q_tot, "q_agents": djc_out["q"]}
