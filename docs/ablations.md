# Ablations

Configs in `gtqn/configs/ablations/` cover attention, reward, and governance ablations.

```bash
python scripts/train.py --config gtqn/configs/default.yaml   gtqn/configs/ablations/attention_dense.yaml
```

(For spatiotemporal ablations, modify `gtqn/models/djc.py`—easy to add as config switches.)
