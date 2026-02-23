# GTQN — Graph Transformer Q-Network (reference implementation)

This repository is a **faithful, engineering-friendly implementation** of the *Graph Transformer Q-Network (GTQN)*
described in the provided manuscript:

> **Graph Transformer Q-Network for Collaborative Governance and Decentralized Decision-Making in Multi-Intersection Networks**

It includes:
- Decentralized **Distributed Junction Controller (DJC)** with **two-stage sparse coordination** (discrete peer gating + soft relevance weighting)
- Centralized Training / Decentralized Execution (**CTDE**) via a **Collaborative Governance Graph (CGG)**
- Unified **spatiotemporal** representation: temporal Transformer + graph-aware coordination
- SUMO/TraCI environment wrapper for multi-intersection control (event-driven, phase-duration actuation)
- Reproducible configs and ablation toggles (attention, reward, governance)

> NOTE: SUMO is external. This repo provides a clean TraCI interface; you must install SUMO and supply `.net.xml` and `.rou.xml` (or generate routes).

---

## Quickstart

### 1) Install SUMO
Ensure `sumo` and `sumo-gui` are in PATH and `SUMO_HOME` is set.

- SUMO docs: https://sumo.dlr.de/docs/
- TraCI traffic light control: https://sumo.dlr.de/docs/TraCI/Change_Traffic_Lights_State.html

### 2) Python env
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Prepare a scenario
Put a SUMO scenario under:
```
gtqn/envs/networks/<scenario_name>/
  - net.net.xml
  - sim.sumocfg
  - routes.rou.xml   (or generate using scripts/make_routes.py)
  - tls_ids.json     (list of controlled traffic light IDs)
  - corridor.json    (optional: corridor lanes + entry junction)
```

### 4) Train
```bash
python scripts/train.py --config gtqn/configs/default.yaml   env.scenario=gtqn/envs/networks/SQ1   train.total_env_steps=1200000
```

### 5) Evaluate
```bash
python scripts/eval.py --run_dir runs/<your_run>
```

---

## Design overview (matches the manuscript)

### DJC (decentralized, executed online)
- Builds a local state embedding `s_i,k` via a **temporal Transformer** over the last `H` decision steps.
- Computes sparse coordination context `c_i,k`:
  1) **Discrete peer gating**: select top-K peers via learned scores (state-dependent)
  2) **Soft relevance weighting**: softmax weights over selected peers
- Outputs per-agent Q-values `Q_i(s_i,k, c_i,k, a)` and selects action ε-greedily.

### CGG (centralized, training-only)
To remain scalable (100 intersections), centralized governance is implemented via a **QMIX-style mixing network**:
- per-agent utilities `q_i`
- mixing network conditioned on global graph embedding → `Q_tot`

This is CTDE: only `q_i` is needed at execution; CGG is used in training to stabilize credit assignment.

---

## License
MIT (see LICENSE).
