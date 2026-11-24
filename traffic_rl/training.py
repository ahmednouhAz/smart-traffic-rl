"""Training loop (rollout collection + PPO updates) and evaluation helpers."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .agents.ppo import PPOAgent, PPOConfig, Trajectory
from .envs.traffic_env import EnvConfig, TrafficNetworkEnv


def run_episode(env: TrafficNetworkEnv, controller, seed: Optional[int] = None, scenario: Optional[str] = None):
    """Play one episode with a controller; return the metrics dict and the queue history."""
    obs = env.reset(seed=seed, scenario=scenario)
    if hasattr(controller, "reset"):
        controller.reset()
    done, ret = False, 0.0
    while not done:
        obs, rewards, done, info = env.step(controller.act(env, obs))
        ret += sum(rewards.values())
    metrics = dict(info["metrics"])
    metrics["return"] = ret
    return metrics, list(env.sim.queue_history)


def evaluate(
    env_cfg: EnvConfig,
    controller,
    scenarios: Iterable[str],
    seeds: Iterable[int] = (100, 101, 102),
) -> List[Dict]:
    rows = []
    for sc in scenarios:
        for s in seeds:
            env = TrafficNetworkEnv(env_cfg, seed=s)
            m, _ = run_episode(env, controller, seed=s, scenario=sc)
            rows.append({"controller": getattr(controller, "name", "?"), "scenario": sc, "seed": s, **m})
    return rows


def train_ppo(
    env_cfg: EnvConfig,
    ppo_cfg: PPOConfig,
    out_dir: Optional[str | Path] = None,
    verbose: bool = True,
    log_every: int = 10,
) -> tuple[PPOAgent, List[Dict]]:
    """Train a shared PPO policy on ``env_cfg`` (any grid size)."""
    env = TrafficNetworkEnv(env_cfg, seed=ppo_cfg.seed)
    agent = PPOAgent(env.observation_dim, env.n_actions, ppo_cfg)
    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    log: List[Dict] = []
    finished: List[tuple] = []  # (trajectory, last_value) waiting for the next update
    n_collected = 0
    best_score = -np.inf
    t0 = time.time()
    last_stats: Dict[str, float] = {}

    for ep in range(ppo_cfg.total_episodes):
        eps = agent.epsilon(ep)
        obs = env.reset(seed=ppo_cfg.seed * 100_000 + ep)
        trajs = {a: Trajectory() for a in env.agents}
        done, ep_return = False, 0.0
        while not done:
            keys = list(obs)
            batch = np.stack([obs[k] for k in keys])
            actions, logps, values = agent.act(batch, epsilon=eps)
            for k, o, a, lp, v in zip(keys, batch, actions, logps, values):
                tr = trajs[k]
                tr.obs.append(o)
                tr.actions.append(int(a))
                tr.logps.append(float(lp))
                tr.values.append(float(v))
            obs, rewards, done, info = env.step({k: int(a) for k, a in zip(keys, actions)})
            for k, r in rewards.items():
                tr = trajs[k]
                if len(tr.rewards) < len(tr.actions):
                    tr.rewards.append(float(r))
                ep_return += r

        # time-limit truncation: bootstrap with the value of the final state
        final_keys = list(obs)
        last_vals = agent.value(np.stack([obs[k] for k in final_keys]))
        for k, lv in zip(final_keys, last_vals):
            finished.append((trajs[k], float(lv)))
            n_collected += len(trajs[k])

        m = info["metrics"]
        row = {
            "episode": ep,
            "scenario": info["scenario"],
            "epsilon": round(eps, 4),
            "return": ep_return,
            "mean_waiting_time": m["mean_waiting_time"],
            "mean_queue_length": m["mean_queue_length"],
            "mean_travel_time": m["mean_travel_time"],
            "throughput": m["throughput"],
            "co2_kg": m["co2_kg"],
        }

        if n_collected >= ppo_cfg.rollout_steps:
            batch = _merge(agent, finished)
            last_stats = agent.update(batch)
            finished, n_collected = [], 0
        row.update({k: round(v, 5) for k, v in last_stats.items()})
        log.append(row)

        # checkpoint on the moving average of the waiting time (lower is better)
        if out and ep >= 20:
            recent = np.mean([r["mean_waiting_time"] for r in log[-20:]])
            if -recent > best_score:
                best_score = -recent
                agent.save(out / "ppo_best.pt", {"episode": ep, "env": _env_dict(env_cfg)})

        if verbose and (ep % log_every == 0 or ep == ppo_cfg.total_episodes - 1):
            recent = log[-log_every:]
            print(
                f"ep {ep:4d} | eps {eps:.2f} | return {np.mean([r['return'] for r in recent]):8.2f} | "
                f"wait {np.mean([r['mean_waiting_time'] for r in recent]):6.2f}s | "
                f"queue {np.mean([r['mean_queue_length'] for r in recent]):6.2f} | "
                f"{time.time() - t0:6.1f}s",
                flush=True,
            )

    if out:
        agent.save(out / "ppo_final.pt", {"episode": ppo_cfg.total_episodes, "env": _env_dict(env_cfg)})
        write_csv(out / "training_log.csv", log)
    return agent, log


def _merge(agent: PPOAgent, finished) -> Dict[str, np.ndarray]:
    obs, acts, logps, advs, rets = [], [], [], [], []
    for traj, last_v in finished:
        n = len(traj)
        if n == 0:
            continue
        adv, ret = agent.compute_gae(traj, last_v)
        obs.append(np.stack(traj.obs[:n]))
        acts.append(np.asarray(traj.actions[:n]))
        logps.append(np.asarray(traj.logps[:n]))
        advs.append(adv)
        rets.append(ret)
    return {
        "obs": np.concatenate(obs),
        "actions": np.concatenate(acts),
        "logps": np.concatenate(logps),
        "advantages": np.concatenate(advs),
        "returns": np.concatenate(rets),
    }


def _env_dict(cfg: EnvConfig) -> dict:
    from dataclasses import asdict

    return asdict(cfg)


def write_csv(path: str | Path, rows: List[Dict]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
