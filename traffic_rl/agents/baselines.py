"""Rule-based traffic-light controllers used as baselines.

Every controller exposes ``act(env, obs) -> {agent: action}`` where ``obs`` is
the dict returned by the environment (only the agents that must decide).
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..envs.simulator import PHASES
from ..envs.traffic_env import TrafficNetworkEnv


class Controller:
    name = "controller"

    def reset(self) -> None:  # pragma: no cover - default no-op
        pass

    def act(self, env: TrafficNetworkEnv, obs: Dict) -> Dict:
        raise NotImplementedError


class FixedTimeController(Controller):
    """Classic pre-timed plan: phases alternate with a constant green time."""

    name = "fixed_time"

    def __init__(self, green: int = 25) -> None:
        self.green = green

    def act(self, env, obs):
        return {a: env.encode_action(1 - env.current_phase(a), self.green) for a in obs}


class LongestQueueController(Controller):
    """Actuated control: serve the phase with the longest queue, green proportional to it."""

    name = "longest_queue"

    def __init__(self, sec_per_vehicle: float = 2.0, g_min: int = 5, g_max: int = 30) -> None:
        self.k, self.g_min, self.g_max = sec_per_vehicle, g_min, g_max

    def act(self, env, obs):
        actions = {}
        for a in obs:
            q = env.lane_queues(a)
            loads = [sum(q[d] for d in lanes) for lanes in PHASES]
            phase = int(np.argmax(loads))
            longest = max(q[d] for d in PHASES[phase])
            green = int(np.clip(self.k * longest, self.g_min, self.g_max))
            actions[a] = env.encode_action(phase, green)
        return actions


class MaxPressureController(Controller):
    """Max-Pressure (Varaiya, 2013): serve the phase with the largest upstream-downstream imbalance."""

    name = "max_pressure"

    def __init__(self, green: int = 10) -> None:
        self.green = green

    def act(self, env, obs):
        actions = {}
        for a in obs:
            q_in, q_out = env.lane_queues(a), env.outgoing_queues(a)
            pressure = [sum(q_in[d] - q_out[d] for d in lanes) for lanes in PHASES]
            actions[a] = env.encode_action(int(np.argmax(pressure)), self.green)
        return actions


class RandomController(Controller):
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, env, obs):
        return {a: int(self.rng.integers(env.n_actions)) for a in obs}


BASELINES = {
    "fixed_time": FixedTimeController,
    "longest_queue": LongestQueueController,
    "max_pressure": MaxPressureController,
    "random": RandomController,
}
