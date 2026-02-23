from __future__ import annotations
from dataclasses import dataclass

@dataclass
class LinearSchedule:
    start: float
    end: float
    frac: float

    def value(self, t: float) -> float:
        if t <= 0:
            return self.start
        if t >= self.frac:
            return self.end
        alpha = t / max(1e-9, self.frac)
        return self.start + alpha * (self.end - self.start)
