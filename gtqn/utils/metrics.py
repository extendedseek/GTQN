from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class VehicleStopTracker:
    stop_speed_threshold: float = 0.1
    prev_moving: Dict[str, bool] = field(default_factory=dict)
    stops: Dict[str, int] = field(default_factory=dict)

    def update(self, veh_id: str, speed: float) -> None:
        moving = speed > self.stop_speed_threshold
        was_moving = self.prev_moving.get(veh_id, True)
        if was_moving and (not moving):
            self.stops[veh_id] = self.stops.get(veh_id, 0) + 1
        self.prev_moving[veh_id] = moving

    def pop_vehicle(self, veh_id: str) -> int:
        self.prev_moving.pop(veh_id, None)
        return self.stops.pop(veh_id, 0)

@dataclass
class EpisodeMetrics:
    total_wait_time: float = 0.0
    total_veh_count: int = 0
    total_stops: int = 0
    queue_sum: float = 0.0
    queue_steps: int = 0
    arrived: int = 0
    sim_seconds: float = 0.0

    def finalize(self) -> Dict[str, float]:
        awt = self.total_wait_time / max(1, self.total_veh_count)
        ans = self.total_stops / max(1, self.total_veh_count)
        aql = self.queue_sum / max(1, self.queue_steps)
        af = self.arrived / max(1e-6, self.sim_seconds)  # veh/s
        return {"AWT": awt, "ANS": ans, "AQL": aql, "AF": af}
