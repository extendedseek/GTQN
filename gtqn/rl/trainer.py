from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim

from gtqn.rl.replay_buffer import ReplayBuffer, to_torch
from gtqn.models.gtqn_system import GTQNSystem
from gtqn.rl.schedules import LinearSchedule

@dataclass
class TrainStats:
    loss: float
    q_tot: float
    grad_norm: float

class Trainer:
    def __init__(
        self,
        model: GTQNSystem,
        target: GTQNSystem,
        replay: ReplayBuffer,
        lr: float,
        lr_final: float,
        lr_decay_frac: float,
        gamma: float,
        target_tau: float,
        grad_clip_norm: float,
        device: str,
    ):
        self.model = model.to(device)
        self.target = target.to(device)
        self.target.load_state_dict(model.state_dict())
        self.replay = replay
        self.gamma = gamma
        self.target_tau = target_tau
        self.grad_clip_norm = grad_clip_norm
        self.device = device

        self.lr_sched = LinearSchedule(lr, lr_final, lr_decay_frac)
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.total_env_steps = 1

    def set_total_env_steps(self, total_steps: int) -> None:
        self.total_env_steps = max(1, int(total_steps))

    @torch.no_grad()
    def soft_update_target(self) -> None:
        for p, tp in zip(self.model.parameters(), self.target.parameters()):
            tp.data.mul_(1.0 - self.target_tau).add_(p.data, alpha=self.target_tau)

    def update_lr(self, env_step: int) -> None:
        t = env_step / float(self.total_env_steps)
        lr = float(self.lr_sched.value(t))
        for g in self.opt.param_groups:
            g["lr"] = lr

    def train_step(self, batch_size: int) -> TrainStats:
        batch = self.replay.sample(batch_size)
        b = to_torch(batch, self.device)

        out = self.model.forward_batch(b["obs_hist"], b["adj"], b["active_mask"], b["act_mask"])
        q_agents = out["q_agents"]  # [B,N,A]

        a = b["actions"].unsqueeze(-1)
        q_taken = torch.gather(q_agents, dim=-1, index=a).squeeze(-1)  # [B,N]
        q_tot_taken = self.model.mix_q(q_taken, out["global_embed"], b["active_mask"], b["act_mask"])  # [B]

        with torch.no_grad():
            next_out_online = self.model.forward_batch(b["next_obs_hist"], b["next_adj"], b["next_active_mask"], b["next_act_mask"])
            next_q_agents_online = next_out_online["q_agents"]
            next_a_star = torch.argmax(next_q_agents_online, dim=-1)

            next_out_target = self.target.forward_batch(b["next_obs_hist"], b["next_adj"], b["next_active_mask"], b["next_act_mask"])
            next_q_agents_target = next_out_target["q_agents"]
            next_q_star = torch.gather(next_q_agents_target, dim=-1, index=next_a_star.unsqueeze(-1)).squeeze(-1)
            next_q_tot = self.target.mix_q(next_q_star, next_out_target["global_embed"], b["next_active_mask"], b["next_act_mask"])

            y = b["reward_net"].view(1).expand_as(next_q_tot) + (1.0 - b["done"].view(-1)) * self.gamma * next_q_tot

        loss = torch.mean((y - q_tot_taken) ** 2)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm))
        self.opt.step()
        self.soft_update_target()

        return TrainStats(loss=float(loss.item()), q_tot=float(q_tot_taken.mean().item()), grad_norm=grad_norm)
