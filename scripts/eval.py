from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import yaml
from tqdm import trange

from gtqn.envs.sumo_env import MultiIntersectionSUMOEnv, EnvConfig
from gtqn.models.gtqn_system import GTQNSystem
from gtqn.utils.checkpoint import load_checkpoint
from gtqn.utils.seed import set_seed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    ckpts = sorted(run_dir.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    ckpt = load_checkpoint(str(ckpts[-1]), map_location="cpu")

    set_seed(int(cfg.get("seed", 0)))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    env_cfg = EnvConfig(
        scenario=str(cfg["env"]["scenario"]),
        use_gui=False,
        sumo_binary=str(cfg["env"]["sumo_binary"]),
        step_length_s=float(cfg["env"]["step_length_s"]),
        max_sim_seconds=float(cfg["env"]["max_sim_seconds"]),
        min_green_s=int(cfg["env"]["min_green_s"]),
        max_green_s=int(cfg["env"]["max_green_s"]),
        amber_s=int(cfg["env"]["amber_s"]),
        decision_every_s=cfg["env"]["decision_every_s"],
        stop_speed_threshold=float(cfg["env"]["stop_speed_threshold"]),
        corridor_enabled=bool(cfg["env"]["corridor"]["enabled"]),
        entry_tls_id=cfg["env"]["corridor"]["entry_tls_id"],
        corridor_in_lanes=cfg["env"]["corridor"]["corridor_in_lanes"],
    )

    dummy_env = MultiIntersectionSUMOEnv(env_cfg, history_len=int(cfg["model"]["history_len"]), obs_dim=int(cfg["model"]["obs_dim"]))
    N_max = dummy_env.N_max
    n_actions = 1 + 4 * len(dummy_env.action_durations)
    dummy_env.close()

    model = GTQNSystem(
        obs_dim=int(cfg["model"]["obs_dim"]),
        history_len=int(cfg["model"]["history_len"]),
        embed_dim=int(cfg["model"]["embed_dim"]),
        n_heads=int(cfg["model"]["n_heads"]),
        n_layers=int(cfg["model"]["n_layers"]),
        dropout=float(cfg["model"]["dropout"]),
        n_actions=n_actions,
        N_max=N_max,
        coord_variant=str(cfg["model"]["coord"]["variant"]),
        K=int(cfg["model"]["coord"]["K"]),
        use_gumbel_topk=bool(cfg["model"]["coord"]["use_gumbel_topk"]),
        cgg_enabled=bool(cfg["model"]["cgg"]["enabled"]),
        mixer=str(cfg["model"]["cgg"]["mixer"]),
        hypernet_embed=int(cfg["model"]["cgg"]["hypernet_embed"]),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    env = MultiIntersectionSUMOEnv(env_cfg, history_len=int(cfg["model"]["history_len"]), obs_dim=int(cfg["model"]["obs_dim"]), seed=int(cfg.get("seed", 0)))
    results = []
    for _ in trange(args.episodes):
        obs = env.reset()
        done = False
        while not done:
            obs_hist = torch.tensor(obs["obs_hist"], device=device, dtype=torch.float32).unsqueeze(0)
            adj = torch.tensor(obs["adj"], device=device, dtype=torch.float32).unsqueeze(0)
            active_mask = torch.tensor(obs["active_mask"], device=device, dtype=torch.float32).unsqueeze(0)
            act_mask = torch.tensor(obs["act_mask"], device=device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q = model.forward_batch(obs_hist, adj, active_mask, act_mask)["q_agents"][0].cpu().numpy()
            actions = np.zeros((env.N_max,), dtype=np.int64)
            for i in range(env.N_max):
                actions[i] = 0 if obs["act_mask"][i] <= 0 else int(np.argmax(q[i]))
            obs, r_local, r_net, done, info = env.step(actions)
        results.append(info.get("episode_metrics") or {})

    env.close()
    out_path = run_dir / "eval_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved:", out_path)
    print("Mean:", {k: float(np.mean([r.get(k, 0.0) for r in results])) for k in ["AWT", "ANS", "AQL", "AF"]})

if __name__ == "__main__":
    main()
