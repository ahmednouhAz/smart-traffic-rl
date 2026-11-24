"""Reinforcement-learning environments built on top of :mod:`simulator`.

``TrafficNetworkEnv`` is a (multi-agent) semi-Markov environment: every
intersection is an agent. An agent is asked for a decision only when its green
light expires, so agents act asynchronously. A single intersection is just the
``1 x 1`` grid.

``SingleIntersectionEnv`` wraps the 1x1 case into a standard Gymnasium ``Env``
(Box observation, Discrete action) so it can be used with any RL library.

State (per intersection, see :meth:`TrafficNetworkEnv.observation_dim`)
    for each of the 4 incoming lanes:
        * number of waiting vehicles (normalised)
        * mean waiting time of the vehicles on the lane (normalised)
        * binary occupancy of the ``obs_cells`` cells closest to the stop line
          (position of the vehicles in the intersection zone)
    for each of the 4 outgoing directions: occupancy of the downstream lane
    one-hot current phase, time since the green started

Action
    ``a = phase * n_durations + k``: the phase that must be green next
    (keeping the current phase extends it, a different phase triggers a yellow
    transition) and its green duration ``green_durations[k]``.

Reward (accumulated between two decisions of an agent)
    ``- wait_coef * (vehicle-seconds of waiting)``
    ``+ queue_coef * (queue before decision - queue after)``
    ``- co2_coef * (grams of CO2 emitted)`` (0 by default)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from .scenarios import Scenario, build_scenarios
from .simulator import DIRECTIONS, HEADING, PHASES, TrafficSimulator

Inter = Tuple[int, int]


@dataclass
class EnvConfig:
    rows: int = 1
    cols: int = 1
    episode_length: int = 3600  # seconds of simulated traffic
    n_cells: int = 25  # lane length = 25 * 7.5 m = 187.5 m
    obs_cells: int = 10  # cells close to the stop line in the observation
    v_max: int = 2
    sat_headway: int = 2
    yellow_time: int = 3
    green_durations: Tuple[int, ...] = (5, 10, 15, 20, 25, 30)
    wait_coef: float = 0.01
    queue_coef: float = 0.1
    co2_coef: float = 0.0
    scenarios: Tuple[str, ...] = ("normal",)
    turn_probs: Tuple[float, float, float] = (0.7, 0.15, 0.15)


class TrafficNetworkEnv:
    """Asynchronous multi-agent environment (one agent per intersection)."""

    def __init__(self, config: Optional[EnvConfig] = None, seed: Optional[int] = None) -> None:
        self.cfg = config or EnvConfig()
        self.sim = TrafficSimulator(
            rows=self.cfg.rows,
            cols=self.cfg.cols,
            n_cells=self.cfg.n_cells,
            v_max=self.cfg.v_max,
            sat_headway=self.cfg.sat_headway,
            yellow_time=self.cfg.yellow_time,
            turn_probs=self.cfg.turn_probs,
            seed=seed,
        )
        self.agents: List[Inter] = list(self.sim.intersections)
        self.all_scenarios: Dict[str, Scenario] = build_scenarios(
            self.cfg.rows, self.cfg.cols, self.cfg.episode_length
        )
        self.scenario: Scenario = self.all_scenarios[self.cfg.scenarios[0]]
        self._rng = np.random.default_rng(seed)
        self._pending: List[Inter] = []
        self._acc_reward: Dict[Inter, float] = {}
        self._queue_at_decision: Dict[Inter, int] = {}
        self._co2_prev = 0.0

    # ----------------------------------------------------------- dimensions
    @property
    def n_actions(self) -> int:
        return len(PHASES) * len(self.cfg.green_durations)

    @property
    def observation_dim(self) -> int:
        per_lane = 2 + self.cfg.obs_cells
        return 4 * per_lane + 4 + len(PHASES) + 1

    def decode_action(self, action: int) -> Tuple[int, int]:
        n_d = len(self.cfg.green_durations)
        phase, k = divmod(int(action), n_d)
        return phase, self.cfg.green_durations[k]

    def encode_action(self, phase: int, duration: int) -> int:
        durations = np.array(self.cfg.green_durations)
        k = int(np.argmin(np.abs(durations - duration)))
        return phase * len(durations) + k

    # ---------------------------------------------------------------- API
    def reset(self, seed: Optional[int] = None, scenario: Optional[str] = None) -> Dict[Inter, np.ndarray]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        name = scenario or self.cfg.scenarios[int(self._rng.integers(len(self.cfg.scenarios)))]
        self.scenario = self.all_scenarios[name]
        self._demand = self.scenario.demand(self.cfg.episode_length)
        self.sim.reset(seed=int(self._rng.integers(2**31 - 1)))
        self._pending = list(self.agents)
        self._acc_reward = {a: 0.0 for a in self.agents}
        self._queue_at_decision = {a: 0 for a in self.agents}
        return {a: self.observe(a) for a in self._pending}

    def step(
        self, actions: Dict[Inter, int]
    ) -> Tuple[Dict[Inter, np.ndarray], Dict[Inter, float], bool, dict]:
        """Apply the actions of the pending agents and simulate until a new decision is needed.

        Returns ``(obs, rewards, done, info)`` where ``obs`` and ``rewards`` are
        keyed by the agents that must act next (all agents when ``done``).
        """
        for a in self._pending:
            phase, duration = self.decode_action(actions[a])
            self.sim.set_signal(a, phase, duration)
            self._acc_reward[a] = 0.0
            self._queue_at_decision[a] = self.sim.intersection_queue(a)

        done = False
        while True:
            self._apply_incidents()
            co2_before = self.sim.co2_g
            expired = self.sim.step(self._demand)
            co2_tick = self.sim.co2_g - co2_before
            n_agents = len(self.agents)
            for a in self.agents:
                self._acc_reward[a] -= self.cfg.wait_coef * self.sim.intersection_waiting(a)
                if self.cfg.co2_coef:
                    self._acc_reward[a] -= self.cfg.co2_coef * co2_tick / n_agents
            if self.sim.t >= self.cfg.episode_length:
                done = True
                break
            if expired:
                break

        self._pending = list(self.agents) if done else expired
        rewards = {}
        for a in self._pending:
            q_now = self.sim.intersection_queue(a)
            rewards[a] = self._acc_reward[a] + self.cfg.queue_coef * (self._queue_at_decision[a] - q_now)
        obs = {a: self.observe(a) for a in self._pending}
        info = {"t": self.sim.t, "scenario": self.scenario.name}
        if done:
            info["metrics"] = self.sim.metrics()
        return obs, rewards, done, info

    def _apply_incidents(self) -> None:
        t = self.sim.t
        for inc in self.scenario.incidents:
            lane = self.sim.lanes.get((inc.inter, inc.origin))
            if lane is None:
                continue
            lane.headway_factor = inc.headway_factor if inc.t_start <= t < inc.t_end else 1.0

    # ---------------------------------------------------------- observation
    def observe(self, inter: Inter) -> np.ndarray:
        sim, k = self.sim, self.cfg.obs_cells
        feats: List[float] = []
        for d in DIRECTIONS:
            lane = sim.lanes[(inter, d)]
            feats.append(min(lane.queue_length() / lane.n_cells, 2.0))
            feats.append(min(sim.lane_mean_wait(lane) / 100.0, 3.0))
            occ = np.zeros(k, dtype=np.float32)
            for v in lane.vehicles:
                if v.pos < k:
                    occ[v.pos] = 1.0
            feats.extend(occ.tolist())
        for d in DIRECTIONS:  # outgoing direction = heading d
            nxt = sim.next_lane(inter, d)
            feats.append(nxt.occupancy() if nxt is not None else 0.0)
        sig = sim.signals[inter]
        current = sig.next_phase if sig.mode == "yellow" else sig.phase
        phase_oh = [0.0] * len(PHASES)
        phase_oh[current] = 1.0
        feats.extend(phase_oh)
        feats.append(min(sig.elapsed / 60.0, 3.0))
        return np.asarray(feats, dtype=np.float32)

    # -------------------------------------------- helpers for rule-based baselines
    def lane_queues(self, inter: Inter) -> Dict[str, int]:
        return {d: self.sim.lanes[(inter, d)].queue_length() for d in DIRECTIONS}

    def outgoing_queues(self, inter: Inter) -> Dict[str, int]:
        """Queue on the lane that vehicles of approach ``d`` enter after going straight."""
        out = {}
        for d in DIRECTIONS:
            nxt = self.sim.next_lane(inter, HEADING[d])
            out[d] = nxt.queue_length() if nxt is not None else 0
        return out

    def current_phase(self, inter: Inter) -> int:
        sig = self.sim.signals[inter]
        return sig.next_phase if sig.mode == "yellow" else sig.phase


class SingleIntersectionEnv(gym.Env):
    """Standard Gymnasium interface for a single intersection."""

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[EnvConfig] = None, seed: Optional[int] = None) -> None:
        cfg = config or EnvConfig()
        cfg.rows, cfg.cols = 1, 1
        self.net = TrafficNetworkEnv(cfg, seed=seed)
        self.agent = self.net.agents[0]
        self.observation_space = gym.spaces.Box(0.0, 3.0, shape=(self.net.observation_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.net.n_actions)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        scenario = (options or {}).get("scenario")
        obs = self.net.reset(seed=seed, scenario=scenario)
        return obs[self.agent], {"scenario": self.net.scenario.name}

    def step(self, action):
        obs, rew, done, info = self.net.step({self.agent: int(action)})
        # The episode ends because of the time limit -> truncated, not terminated.
        return obs[self.agent], float(rew[self.agent]), False, done, info
