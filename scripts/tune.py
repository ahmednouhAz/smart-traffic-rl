"""Random-search hyper-parameter optimisation for PPO.

Each trial trains a short PPO run on the single intersection and is scored on
the mean waiting time over the training scenarios (lower is better).

    python scripts/tune.py --trials 12 --episodes 150 --out results/tuning
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from traffic_rl.agents.ppo import PPOConfig, PPOController  # noqa: E402
from traffic_rl.envs import TRAIN_SCENARIOS, EnvConfig  # noqa: E402
from traffic_rl.training import evaluate, train_ppo, write_csv  # noqa: E402

SEARCH_SPACE = {
    "lr": [1e-4, 3e-4, 1e-3],
    "gamma": [0.9, 0.95, 0.98, 0.99],
    "clip_eps": [0.1, 0.2, 0.3],
    "entropy_coef": [0.0, 0.01, 0.03],
    "hidden": [64, 128, 256],
    "eps_start": [0.0, 0.3, 0.5],
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=12)
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/tuning")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env_cfg = EnvConfig(scenarios=TRAIN_SCENARIOS)
    results = []
    for trial in range(args.trials):
        params = {k: v[int(rng.integers(len(v)))] for k, v in SEARCH_SPACE.items()}
        cfg = PPOConfig(total_episodes=args.episodes, seed=trial, **params)
        agent, _ = train_ppo(env_cfg, cfg, verbose=False)
        rows = evaluate(env_cfg, PPOController(agent), TRAIN_SCENARIOS, seeds=(200, 201))
        score = float(np.mean([r["mean_waiting_time"] for r in rows]))
        results.append({"trial": trial, **params, "mean_waiting_time": round(score, 3)})
        print(f"trial {trial:2d} | wait {score:6.2f}s | {params}", flush=True)

    results.sort(key=lambda r: r["mean_waiting_time"])
    write_csv(out / "tuning_results.csv", results)
    best = results[0]
    with open(out / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    print("\nBest configuration:", json.dumps(best, indent=2))
    print("Default PPOConfig:", {k: asdict(PPOConfig())[k] for k in SEARCH_SPACE})


if __name__ == "__main__":
    main()
