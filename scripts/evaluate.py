"""Compare the trained PPO agent with rule-based controllers on every scenario.

Example
-------
    python scripts/evaluate.py --model results/single/ppo_best.pt --out results/single
    python scripts/evaluate.py --model results/grid2x2/ppo_best.pt --rows 2 --cols 2 --out results/grid2x2

Outputs (in ``--out``): ``eval_metrics.csv`` (one row per run), ``eval_summary.md``
(mean over seeds + improvement vs fixed-time), bar charts per metric, queue
time-series for the hardest scenarios and a snapshot of the network.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from traffic_rl.agents.baselines import (  # noqa: E402
    FixedTimeController,
    LongestQueueController,
    MaxPressureController,
    RandomController,
)
from traffic_rl.agents.ppo import PPOAgent, PPOController  # noqa: E402
from traffic_rl.envs import ALL_SCENARIOS, EnvConfig, TrafficNetworkEnv  # noqa: E402
from traffic_rl.training import evaluate, run_episode, write_csv  # noqa: E402
from traffic_rl.utils import plotting  # noqa: E402

METRICS = {
    "mean_waiting_time": "Mean waiting time (s/veh)",
    "mean_travel_time": "Mean travel time (s)",
    "mean_queue_length": "Mean number of waiting vehicles",
    "throughput": "Vehicles served",
    "stops_per_vehicle": "Stops per vehicle",
    "co2_kg": "CO2 emitted (kg)",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--rows", type=int, default=1)
    p.add_argument("--cols", type=int, default=1)
    p.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    p.add_argument("--scenarios", nargs="+", default=list(ALL_SCENARIOS))
    p.add_argument("--out", default="results/single")
    p.add_argument("--no-random", action="store_true", help="skip the random baseline")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env_cfg = EnvConfig(rows=args.rows, cols=args.cols)
    controllers = [
        PPOController(PPOAgent.load(args.model)),
        FixedTimeController(),
        LongestQueueController(),
        MaxPressureController(),
    ]
    if not args.no_random:
        controllers.append(RandomController())

    rows = []
    for c in controllers:
        print(f"evaluating {c.name} ...", flush=True)
        rows += evaluate(env_cfg, c, args.scenarios, args.seeds)
    write_csv(out / "eval_metrics.csv", rows)

    # mean over seeds
    summary = defaultdict(lambda: defaultdict(dict))
    names = [c.name for c in controllers]
    for c in names:
        for sc in args.scenarios:
            sel = [r for r in rows if r["controller"] == c and r["scenario"] == sc]
            for m in METRICS:
                summary[c][sc][m] = float(np.mean([r[m] for r in sel]))

    lines = [f"# Evaluation ({args.rows}x{args.cols} grid, {len(args.seeds)} seeds per scenario)\n"]
    for m, label in METRICS.items():
        lines.append(f"\n## {label}\n")
        lines.append("| scenario | " + " | ".join(names) + " | PPO vs fixed-time |")
        lines.append("|---" * (len(names) + 2) + "|")
        for sc in args.scenarios:
            vals = [summary[c][sc][m] for c in names]
            base = summary["fixed_time"][sc][m]
            gain = (summary["ppo"][sc][m] - base) / base * 100 if base else 0.0
            lines.append(f"| {sc} | " + " | ".join(f"{v:.2f}" for v in vals) + f" | {gain:+.1f}% |")
        plotting.plot_metric_bars(summary, m, args.scenarios, names, out / f"eval_{m}.png", label)
    (out / "eval_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # queue time-series on two hard scenarios, same seed for every controller
    for sc in ("rush_hour", "accident"):
        if sc not in args.scenarios:
            continue
        hist = {}
        for c in controllers:
            if c.name == "random":
                continue
            env = TrafficNetworkEnv(env_cfg, seed=args.seeds[0])
            _, hist[c.name] = run_episode(env, c, seed=args.seeds[0], scenario=sc)
        plotting.plot_queue_timeseries(hist, out / f"queue_{sc}.png", f"Waiting vehicles over time - {sc}")

    # snapshot of the network controlled by PPO in the middle of the rush hour
    env = TrafficNetworkEnv(env_cfg, seed=args.seeds[0])
    obs = env.reset(seed=args.seeds[0], scenario="rush_hour")
    while env.sim.t < env_cfg.episode_length // 2:
        obs, _, done, _ = env.step(controllers[0].act(env, obs))
        if done:
            break
    plotting.render_network(env.sim, out / "snapshot_rush_hour.png", f"PPO, rush hour, t = {env.sim.t} s")
    print(f"\nSaved results to {out}/")


if __name__ == "__main__":
    main()
