from __future__ import annotations
import argparse, time
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch
import yaml
from tqdm import tqdm

from gtqn.utils.seed import set_seed
from gtqn.utils.logger import Logger
from gtqn.utils.checkpoint import save_checkpoint
from gtqn.envs.sumo_env import MultiIntersectionSUMOEnv, EnvConfig
from gtqn.rl.replay_buffer import ReplayBuffer, Transition
from gtqn.rl.trainer import Trainer
from gtqn.rl.schedules import LinearSchedule
from gtqn.models.gtqn_system import GTQNSystem

def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d

def parse_overrides(args_list):
    out = {}
    for s in args_list:
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        if v.lower() in ["true", "false"]:
            vv = v.lower() == "true"
        elif v.lower() in ["null", "none"]:
            vv = None
        else:
            try:
                vv = int(v)
            except ValueError:
                try:
                    vv = float(v)
                except ValueError:
                    vv = v
        cur = out
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = vv
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = deep_update(cfg, parse_overrides(args.overrides))

    set_seed(int(cfg.get("seed", 0)))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = Path("runs") / time.strftime("gtqn_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    logger = Logger.create(str(run_dir))

    env_cfg = EnvConfig(
        scenario=str(cfg["env"]["scenario"]),
        use_gui=bool(cfg["env"]["use_gui"]),
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
    n_actions = 1 + 4 * len(dummy_env.action_durations)  # NOOP + phases*durations
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
    )
    target = GTQNSystem(
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
    )

    replay = ReplayBuffer(int(cfg["train"]["replay_size"]), seed=int(cfg.get("seed", 0)))
    trainer = Trainer(
        model=model, target=target, replay=replay,
        lr=float(cfg["train"]["lr"]), lr_final=float(cfg["train"]["lr_final"]), lr_decay_frac=float(cfg["train"]["lr_decay_frac"]),
        gamma=float(cfg["train"]["gamma"]), target_tau=float(cfg["train"]["target_tau"]),
        grad_clip_norm=float(cfg["train"]["grad_clip_norm"]),
        device=device,
    )
    trainer.set_total_env_steps(int(cfg["train"]["total_env_steps"]))
    eps_sched = LinearSchedule(float(cfg["train"]["epsilon_start"]), float(cfg["train"]["epsilon_final"]), float(cfg["train"]["epsilon_decay_frac"]))

    env = MultiIntersectionSUMOEnv(env_cfg, history_len=int(cfg["model"]["history_len"]), obs_dim=int(cfg["model"]["obs_dim"]), seed=int(cfg.get("seed", 0)))
    obs = env.reset()
    env_step = 0
    episode = 0

    pbar = tqdm(total=int(cfg["train"]["total_env_steps"]))
    while env_step < int(cfg["train"]["total_env_steps"]):
        eps = float(eps_sched.value(env_step / float(cfg["train"]["total_env_steps"])))
        trainer.update_lr(env_step)

        obs_hist = torch.tensor(obs["obs_hist"], device=device, dtype=torch.float32).unsqueeze(0)
        adj = torch.tensor(obs["adj"], device=device, dtype=torch.float32).unsqueeze(0)
        active_mask = torch.tensor(obs["active_mask"], device=device, dtype=torch.float32).unsqueeze(0)
        act_mask = torch.tensor(obs["act_mask"], device=device, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            out = trainer.model.forward_batch(obs_hist, adj, active_mask, act_mask)
            q = out["q_agents"][0].cpu().numpy()

        actions = np.zeros((env.N_max,), dtype=np.int64)
        for i in range(env.N_max):
            if obs["act_mask"][i] <= 0:
                actions[i] = 0
                continue
            if np.random.rand() < eps:
                actions[i] = np.random.randint(1, n_actions)
            else:
                actions[i] = int(np.argmax(q[i]))

        next_obs, r_local, r_net, done, info = env.step(actions)

        replay.add(Transition(
            obs_hist=obs["obs_hist"],
            adj=obs["adj"],
            active_mask=obs["active_mask"],
            act_mask=obs["act_mask"],
            actions=actions,
            reward_local=r_local,
            reward_net=float(r_net),
            next_obs_hist=next_obs["obs_hist"],
            next_adj=next_obs["adj"],
            next_active_mask=next_obs["active_mask"],
            next_act_mask=next_obs["act_mask"],
            done=float(done),
        ))

        obs = next_obs
        env_step += 1
        pbar.update(1)

        if env_step > int(cfg["train"]["warmup_steps"]) and (env_step % int(cfg["train"]["train_every"]) == 0) and len(replay) >= int(cfg["train"]["batch_size"]):
            stats = trainer.train_step(int(cfg["train"]["batch_size"]))
            logger.log(env_step, {"train/loss": stats.loss, "train/q_tot": stats.q_tot, "train/grad_norm": stats.grad_norm, "train/epsilon": eps})

        if done:
            episode += 1
            em = info.get("episode_metrics") or {}
            if em:
                logger.log(env_step, {f"episode/{k}": v for k, v in em.items()})
            obs = env.reset()

        if env_step % int(cfg["train"]["save_every"]) == 0:
            save_checkpoint(str(run_dir / f"ckpt_{env_step}.pt"), {"env_step": env_step, "episode": episode, "model": trainer.model.state_dict(), "cfg": cfg})

    pbar.close()
    env.close()
    logger.close()
    print(f"Done. Run directory: {run_dir}")

if __name__ == "__main__":
    main()
