from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict
from torch.utils.tensorboard import SummaryWriter

@dataclass
class Logger:
    run_dir: str
    tb: SummaryWriter

    @classmethod
    def create(cls, run_dir: str) -> "Logger":
        os.makedirs(run_dir, exist_ok=True)
        return cls(run_dir=run_dir, tb=SummaryWriter(run_dir))

    def log(self, step: int, metrics: Dict[str, Any]) -> None:
        for k, v in metrics.items():
            if v is None:
                continue
            try:
                self.tb.add_scalar(k, float(v), step)
            except Exception:
                pass

    def close(self) -> None:
        self.tb.flush()
        self.tb.close()
