from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import json
import numpy as np
from pathlib import Path

import traci
import sumolib  # type: ignore

from gtqn.envs.obs_builder import ObsBuilder, ObsBuilderConfig
from gtqn.utils.metrics import EpisodeMetrics, VehicleStopTracker

@dataclass
class EnvConfig:
    scenario: str
    use_gui: bool
    sumo_binary: str
    step_length_s: float
    max_sim_seconds: float
    min_green_s: int
    max_green_s: int
    amber_s: int
    decision_every_s: Optional[int]
    stop_speed_threshold: float
    corridor_enabled: bool = True
    entry_tls_id: Optional[str] = None
    corridor_in_lanes: Optional[List[str]] = None

class MultiIntersectionSUMOEnv:
    def __init__(
        self,
        cfg: EnvConfig,
        history_len: int,
        obs_dim: int,
        action_durations: List[int] = [10, 20, 30, 40, 50, 60],
        seed: int = 0,
    ):
        self.cfg = cfg
        self.history_len = history_len
        self.obs_dim = obs_dim
        self.action_durations = [d for d in action_durations if cfg.min_green_s <= d <= cfg.max_green_s]
        self.rng = np.random.default_rng(seed)

        self.scenario_dir = Path(cfg.scenario)
        self.sumocfg = str(self.scenario_dir / "sim.sumocfg")
        self.tls_ids = self._load_tls_ids()
        self.N = len(self.tls_ids)
        self.N_max = self.N

        self.adj = self._build_adjacency_from_net()
        self.active_mask = np.ones((self.N_max,), dtype=np.float32)

        self.obs_builder = ObsBuilder(ObsBuilderConfig(obs_dim=obs_dim))
        self.obs_hist = np.zeros((self.N_max, history_len, obs_dim), dtype=np.float32)

        self.phase_idx = np.zeros((self.N_max,), dtype=np.int64)
        self.remaining = np.zeros((self.N_max,), dtype=np.float32)
        self.elapsed = np.zeros((self.N_max,), dtype=np.float32)

        self.metrics = EpisodeMetrics()
        self.stop_tracker = VehicleStopTracker(stop_speed_threshold=cfg.stop_speed_threshold)

        self.entry_tls_id, self.corridor_in_lanes = self._load_corridor_meta()

        self._sim_time = 0.0
        self._started = False

    def _load_tls_ids(self) -> List[str]:
        p = self.scenario_dir / "tls_ids.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        net_files = list(self.scenario_dir.glob("*.net.xml")) + list(self.scenario_dir.glob("net.net.xml"))
        if not net_files:
            raise FileNotFoundError("No tls_ids.json and no .net.xml found in scenario.")
        net = sumolib.net.readNet(str(net_files[0]))
        tls = [n.getID() for n in net.getNodes() if n.getType() == "traffic_light"]
        tls.sort()
        if not tls:
            raise RuntimeError("No traffic light nodes found.")
        return tls

    def _load_corridor_meta(self) -> Tuple[Optional[str], List[str]]:
        p = self.scenario_dir / "corridor.json"
        if p.exists():
            meta = json.loads(p.read_text(encoding="utf-8"))
            return meta.get("entry_tls_id"), meta.get("corridor_in_lanes", [])
        return self.cfg.entry_tls_id, (self.cfg.corridor_in_lanes or [])

    def _build_adjacency_from_net(self) -> np.ndarray:
        net_files = list(self.scenario_dir.glob("*.net.xml")) + list(self.scenario_dir.glob("net.net.xml"))
        if not net_files:
            raise FileNotFoundError("No .net.xml found in scenario.")
        net = sumolib.net.readNet(str(net_files[0]))
        idx = {tid: i for i, tid in enumerate(self.tls_ids)}
        adj = np.zeros((self.N, self.N), dtype=np.float32)
        for e in net.getEdges():
            fr = e.getFromNode().getID()
            to = e.getToNode().getID()
            if fr in idx and to in idx:
                adj[idx[fr], idx[to]] = 1.0
                adj[idx[to], idx[fr]] = 1.0
        np.fill_diagonal(adj, 1.0)
        return adj

    def _start_sumo(self) -> None:
        if self._started:
            return
        sumo_bin = self.cfg.sumo_binary
        if self.cfg.use_gui:
            sumo_bin = sumo_bin.replace("sumo", "sumo-gui")
        cmd = [
            sumo_bin,
            "-c", self.sumocfg,
            "--step-length", str(self.cfg.step_length_s),
            "--no-step-log", "true",
            "--time-to-teleport", "-1",
        ]
        traci.start(cmd)
        self._started = True

    def close(self) -> None:
        if self._started:
            traci.close()
        self._started = False

    def reset(self) -> Dict[str, Any]:
        self.close()
        self._start_sumo()
        self._sim_time = 0.0

        self.obs_hist[:] = 0.0
        self.phase_idx[:] = 0
        self.remaining[:] = float(self.cfg.min_green_s)
        self.elapsed[:] = 0.0
        self.metrics = EpisodeMetrics()
        self.stop_tracker = VehicleStopTracker(stop_speed_threshold=self.cfg.stop_speed_threshold)

        for i, tid in enumerate(self.tls_ids):
            traci.trafficlight.setPhase(tid, int(self.phase_idx[i]))
            traci.trafficlight.setPhaseDuration(tid, float(self.remaining[i]))

        self._step_sim(1)
        self._update_obs_hist()
        act_mask = self._compute_act_mask()
        return self._obs_dict(act_mask)

    def _obs_dict(self, act_mask: np.ndarray) -> Dict[str, Any]:
        return {
            "obs_hist": self.obs_hist.copy(),
            "adj": self.adj.copy(),
            "active_mask": self.active_mask.copy(),
            "act_mask": act_mask.astype(np.float32),
            "tls_ids": list(self.tls_ids),
            "time": float(self._sim_time),
        }

    def _step_sim(self, n: int) -> None:
        for _ in range(n):
            traci.simulationStep()
            self._sim_time = float(traci.simulation.getTime())
            veh_ids = traci.vehicle.getIDList()
            for vid in veh_ids:
                sp = traci.vehicle.getSpeed(vid)
                self.stop_tracker.update(vid, sp)
                self.metrics.total_wait_time += float(traci.vehicle.getWaitingTime(vid))
            self.metrics.queue_sum += float(self._network_queue())
            self.metrics.queue_steps += 1
            self.metrics.arrived += int(traci.simulation.getArrivedNumber())
            self.metrics.sim_seconds = self._sim_time
            if self._sim_time >= self.cfg.max_sim_seconds:
                break

    def _network_queue(self) -> float:
        q = 0.0
        for tid in self.tls_ids:
            lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tid)))
            for ln in lanes:
                q += traci.lane.getLastStepHaltingNumber(ln)
        return q

    def _intersection_summaries(self, tid: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tid)))
        buckets = [lanes[i::4] for i in range(4)]  # heuristic lane->dir split
        queues = np.zeros(4, dtype=np.float32)
        delay = np.zeros(4, dtype=np.float32)
        occ = np.zeros(4, dtype=np.float32)
        speed = np.zeros(4, dtype=np.float32)
        for d in range(4):
            lns = buckets[d]
            if not lns:
                continue
            qd = wd = od = sd = 0.0
            for ln in lns:
                qd += traci.lane.getLastStepHaltingNumber(ln)
                od += traci.lane.getLastStepOccupancy(ln) / 100.0
                sd += traci.lane.getLastStepMeanSpeed(ln) / 13.9
                vids = traci.lane.getLastStepVehicleIDs(ln)
                for v in vids:
                    wd += traci.vehicle.getWaitingTime(v)
            queues[d] = qd
            delay[d] = wd / max(1, len(lns))
            occ[d] = od / max(1, len(lns))
            speed[d] = sd / max(1, len(lns))
        return queues, delay, occ, speed

    def _update_obs_hist(self) -> None:
        self.obs_hist = np.roll(self.obs_hist, shift=-1, axis=1)
        for i, tid in enumerate(self.tls_ids):
            queues, delay, occ, speed = self._intersection_summaries(tid)
            obs = self.obs_builder.build(
                phase_idx=int(self.phase_idx[i]),
                remaining_green=float(self.remaining[i]),
                elapsed_green=float(self.elapsed[i]),
                queues_by_dir=queues,
                delay_by_dir=delay,
                occ_by_dir=occ,
                speed_by_dir=speed,
            )
            self.obs_hist[i, -1, :] = obs

    def _compute_act_mask(self) -> np.ndarray:
        return (self.remaining <= 0.0).astype(np.float32)

    def step(self, actions: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray, float, bool, Dict[str, Any]]:
        act_mask = self._compute_act_mask()
        for i, tid in enumerate(self.tls_ids):
            if act_mask[i] <= 0:
                self.elapsed[i] += self.cfg.step_length_s
                self.remaining[i] -= self.cfg.step_length_s
                continue
            a = int(actions[i])
            if a <= 0:
                a = int(self.rng.integers(1, 1 + 4 * len(self.action_durations)))
            a -= 1
            phase = int(a // len(self.action_durations)) % 4
            dur = float(self.action_durations[int(a % len(self.action_durations))])
            self.phase_idx[i] = phase
            self.elapsed[i] = 0.0
            self.remaining[i] = dur
            traci.trafficlight.setPhase(tid, phase)
            traci.trafficlight.setPhaseDuration(tid, dur)

        if self.cfg.decision_every_s is not None:
            self._step_sim(int(self.cfg.decision_every_s))
            for i in range(self.N):
                self.elapsed[i] += self.cfg.decision_every_s
                self.remaining[i] -= self.cfg.decision_every_s
        else:
            while True:
                self._step_sim(1)
                for i in range(self.N):
                    self.elapsed[i] += self.cfg.step_length_s
                    self.remaining[i] -= self.cfg.step_length_s
                if self._sim_time >= self.cfg.max_sim_seconds:
                    break
                if (self.remaining <= 0.0).any():
                    break

        self._update_obs_hist()
        next_act_mask = self._compute_act_mask()

        r_local, r_net = self._compute_rewards()
        done = bool(self._sim_time >= self.cfg.max_sim_seconds or traci.simulation.getMinExpectedNumber() <= 0)

        if done:
            in_net = traci.vehicle.getIDCount()
            self.metrics.total_veh_count = int(self.metrics.arrived + in_net)
            for vid in traci.vehicle.getIDList():
                self.metrics.total_stops += self.stop_tracker.pop_vehicle(vid)

        info = {"time": self._sim_time, "episode_metrics": (self.metrics.finalize() if done else None)}
        return self._obs_dict(next_act_mask), r_local, float(r_net), done, info

    def _compute_rewards(self) -> Tuple[np.ndarray, float]:
        r_local = np.zeros((self.N_max,), dtype=np.float32)
        hind = np.zeros((self.N_max,), dtype=np.float32)
        for i, tid in enumerate(self.tls_ids):
            queues, delay, _, _ = self._intersection_summaries(tid)
            r_local[i] = -(queues.sum() + delay.sum())
            if self.corridor_in_lanes and (self.entry_tls_id is None or tid != self.entry_tls_id):
                h = 0.0
                for ln in self.corridor_in_lanes:
                    try:
                        h += traci.lane.getLastStepHaltingNumber(ln)
                    except traci.exceptions.TraCIException:
                        continue
                hind[i] = h
        r_net = float(np.mean(r_local - hind))
        return r_local, r_net
