"""Train a PPO traffic-light controller.

Examples
--------
Single intersection (default):
    python scripts/train.py --episodes 600 --out results/single

2x2 network of intersections (shared policy):
    python scripts/train.py --rows 2 --cols 2 --episodes 400 --out results/grid2x2
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from traffic_rl.agents.ppo import PPOConfig  # noqa: E402
from traffic_rl.envs import TRAIN_SCENARIOS, EnvConfig  # noqa: E402
from traffic_rl.training import train_ppo  # noqa: E402
from traffic_rl.utils.plotting import plot_training_curve  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", type=int, default=1)
    p.add_argument("--cols", type=int, default=1)
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--episode-length", type=int, default=3600, help="simulated seconds per episode")
    p.add_argument("--scenarios", nargs="+", default=list(TRAIN_SCENARIOS))
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entropy", type=float, default=0.01)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--rollout", type=int, default=2048)
    p.add_argument("--eps-start", type=float, default=0.3, help="initial epsilon-greedy exploration")
    p.add_argument("--co2-coef", type=float, default=0.0, help="weight of the CO2 penalty in the reward")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="results/single")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env_cfg = EnvConfig(
        rows=args.rows,
        cols=args.cols,
        episode_length=args.episode_length,
        scenarios=tuple(args.scenarios),
        co2_coef=args.co2_coef,
    )
    ppo_cfg = PPOConfig(
        lr=args.lr,
        gamma=args.gamma,
        clip_eps=args.clip,
        entropy_coef=args.entropy,
        hidden=args.hidden,
        rollout_steps=args.rollout,
        eps_start=args.eps_start,
        total_episodes=args.episodes,
        seed=args.seed,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump({"env": asdict(env_cfg), "ppo": asdict(ppo_cfg)}, f, indent=2)

    print(f"Training PPO on a {args.rows}x{args.cols} grid, scenarios={args.scenarios}")
    _, log = train_ppo(env_cfg, ppo_cfg, out_dir=out)
    plot_training_curve(log, out / "training_curve.png")
    print(f"Saved model, log and training curve to {out}/")


if __name__ == "__main__":
    main()
