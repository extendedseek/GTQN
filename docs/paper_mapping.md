# Mapping the manuscript to this implementation

The manuscript defines a heterogeneous graph snapshot `(P_k, C_k, m_k^(i))` with:
- Junction, traffic-stream, and road/link nodes
- Two-stage sparse coordination among intersections
- Temporal Transformer for delayed upstream→downstream effects
- CTDE via centralized governance

This repo implements the **core GTQN mechanisms** with a compact per-intersection observation vector for clarity.
You can extend `ObsBuilder` to produce the full heterogeneous `P_k` exactly as Table-2 and then pool it to
per-intersection tokens.
