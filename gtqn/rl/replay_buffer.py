from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import random
import numpy as np
import torch

@dataclass
class Transition:
    obs_hist: np.ndarray
    adj: np.ndarray
    active_mask: np.ndarray
    act_mask: np.ndarray
    actions: np.ndarray
    reward_local: np.ndarray
    reward_net: float
    next_obs_hist: np.ndarray
    next_adj: np.ndarray
    next_active_mask: np.ndarray
    next_act_mask: np.ndarray
    done: float

class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = int(capacity)
        self.rng = random.Random(seed)
        self.buf: List[Transition] = []
        self.pos = 0

    def __len__(self) -> int:
        return len(self.buf)

    def add(self, tr: Transition) -> None:
        if len(self.buf) < self.capacity:
            self.buf.append(tr)
        else:
            self.buf[self.pos] = tr
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Transition:
        batch = self.rng.sample(self.buf, k=batch_size)

        def stack(attr: str):
            return np.stack([getattr(b, attr) for b in batch], axis=0)

        return Transition(
            obs_hist=stack("obs_hist"),
            adj=stack("adj"),
            active_mask=stack("active_mask"),
            act_mask=stack("act_mask"),
            actions=stack("actions"),
            reward_local=stack("reward_local"),
            reward_net=float(np.mean([b.reward_net for b in batch])),
            next_obs_hist=stack("next_obs_hist"),
            next_adj=stack("next_adj"),
            next_active_mask=stack("next_active_mask"),
            next_act_mask=stack("next_act_mask"),
            done=stack("done"),
        )

def to_torch(batch: Transition, device: str) -> Dict[str, torch.Tensor]:
    def t(x, dtype=None):
        tt = torch.as_tensor(x, device=device)
        if dtype is not None:
            tt = tt.to(dtype)
        return tt

    return {
        "obs_hist": t(batch.obs_hist, torch.float32),
        "adj": t(batch.adj, torch.float32),
        "active_mask": t(batch.active_mask, torch.float32),
        "act_mask": t(batch.act_mask, torch.float32),
        "actions": t(batch.actions, torch.long),
        "reward_local": t(batch.reward_local, torch.float32),
        "reward_net": torch.tensor(batch.reward_net, device=device, dtype=torch.float32),
        "next_obs_hist": t(batch.next_obs_hist, torch.float32),
        "next_adj": t(batch.next_adj, torch.float32),
        "next_active_mask": t(batch.next_active_mask, torch.float32),
        "next_act_mask": t(batch.next_act_mask, torch.float32),
        "done": t(batch.done, torch.float32),
    }
