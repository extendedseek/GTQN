from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class ObsBuilderConfig:
    obs_dim: int = 32

class ObsBuilder:
    """
    Compact per-intersection observation vector (engineering-friendly).
    """
    def __init__(self, cfg: ObsBuilderConfig):
        self.cfg = cfg

    def build(
        self,
        phase_idx: int,
        remaining_green: float,
        elapsed_green: float,
        queues_by_dir: np.ndarray,
        delay_by_dir: np.ndarray,
        occ_by_dir: np.ndarray,
        speed_by_dir: np.ndarray,
    ) -> np.ndarray:
        x = []
        onehot = np.zeros(4, dtype=np.float32)
        onehot[int(phase_idx) % 4] = 1.0
        x.extend(onehot.tolist())
        x.append(float(remaining_green))
        x.append(float(elapsed_green))
        x.append(float(queues_by_dir.sum()))
        x.append(float(delay_by_dir.sum()))
        x.extend(queues_by_dir.astype(np.float32).tolist())
        x.extend(delay_by_dir.astype(np.float32).tolist())
        x.extend(occ_by_dir.astype(np.float32).tolist())
        x.extend(speed_by_dir.astype(np.float32).tolist())
        x = np.asarray(x, dtype=np.float32)
        if x.size < self.cfg.obs_dim:
            x = np.pad(x, (0, self.cfg.obs_dim - x.size))
        return x[: self.cfg.obs_dim]
