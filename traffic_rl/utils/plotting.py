"""Figures for the report / presentation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..envs.simulator import DIRECTIONS, PHASES  # noqa: E402

COLORS = {
    "ppo": "#2a6fdb",
    "fixed_time": "#9aa0a6",
    "longest_queue": "#e8a33d",
    "max_pressure": "#3aa676",
    "random": "#d9534f",
}


def _color(name: str) -> str:
    return COLORS.get(name, "#555555")


def plot_training_curve(log: List[Dict], path: str | Path, window: int = 20) -> None:
    ep = np.array([r["episode"] for r in log])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in zip(
        axes,
        ("return", "mean_waiting_time", "mean_queue_length"),
        ("Episode return", "Mean waiting time (s/veh)", "Mean queue length (veh)"),
    ):
        y = np.array([float(r[key]) for r in log])
        ax.plot(ep, y, color="#2a6fdb", alpha=0.25, lw=1)
        if len(y) >= window:
            smooth = np.convolve(y, np.ones(window) / window, mode="valid")
            ax.plot(ep[window - 1 :], smooth, color="#2a6fdb", lw=2, label=f"moving avg ({window})")
            ax.legend()
        ax.set_title(title)
        ax.set_xlabel("episode")
        ax.grid(alpha=0.3)
    eps = [float(r["epsilon"]) for r in log]
    ax2 = axes[0].twinx()
    ax2.plot(ep, eps, color="#d9534f", ls="--", lw=1)
    ax2.set_ylabel("epsilon (exploration)", color="#d9534f")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_metric_bars(
    summary: Dict[str, Dict[str, Dict[str, float]]],
    metric: str,
    scenarios: Sequence[str],
    controllers: Sequence[str],
    path: str | Path,
    ylabel: str,
) -> None:
    """``summary[controller][scenario][metric]`` -> grouped bar chart."""
    x = np.arange(len(scenarios))
    w = 0.8 / len(controllers)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, c in enumerate(controllers):
        vals = [summary[c][s][metric] for s in scenarios]
        ax.bar(x + (i - (len(controllers) - 1) / 2) * w, vals, w, label=c, color=_color(c))
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " per scenario")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=len(controllers), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_queue_timeseries(histories: Dict[str, List[int]], path: str | Path, title: str, smooth: int = 60) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    for name, h in histories.items():
        y = np.convolve(np.asarray(h, dtype=float), np.ones(smooth) / smooth, mode="valid")
        ax.plot(np.arange(len(y)) / 60.0, y, label=name, color=_color(name), lw=2)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("vehicles waiting in the network")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render_network(sim, path: str | Path, title: str = "") -> None:
    """Snapshot of the network: roads, vehicles (red = stopped) and green approaches."""
    n = sim.n_cells
    spacing = 2 * n + 4
    fig, ax = plt.subplots(figsize=(4 + 3 * sim.cols, 4 + 3 * sim.rows))
    offsets = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}  # unit vector towards the origin
    for (r, c) in sim.intersections:
        cx, cy = c * spacing, -r * spacing
        ax.add_patch(plt.Rectangle((cx - 1, cy - 1), 2, 2, color="#444444"))
        sig = sim.signals[(r, c)]
        for d in DIRECTIONS:
            ux, uy = offsets[d]
            ax.plot([cx + ux, cx + ux * (n + 1)], [cy + uy, cy + uy * (n + 1)], color="#cccccc", lw=8, zorder=0)
            green = sig.mode == "green" and d in PHASES[sig.phase]
            col = "#2ecc71" if green else ("#f1c40f" if sig.mode == "yellow" else "#e74c3c")
            ax.scatter([cx + ux * 1.6], [cy + uy * 1.6], s=60, color=col, edgecolor="black", zorder=3)
            lane = sim.lanes[((r, c), d)]
            xs = [cx + ux * (v.pos + 2) for v in lane.vehicles]
            ys = [cy + uy * (v.pos + 2) for v in lane.vehicles]
            cs = ["#c0392b" if v.stopped else "#2c3e50" for v in lane.vehicles]
            ax.scatter(xs, ys, s=18, c=cs, marker="s", zorder=2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title or f"t = {sim.t} s")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
