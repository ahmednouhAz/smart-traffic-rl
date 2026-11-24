"""Lightweight microscopic traffic simulator (cellular automaton).

The road network is a grid of signalised intersections. Every intersection has
four *incoming* lanes, one per approach (N, E, S, W). A lane is split into
``n_cells`` cells of ``CELL_LENGTH`` metres; cell 0 is the stop line. Time
advances in steps of one second.

Movement rules (simplified Nagel-Schreckenberg model):
* a vehicle advances up to ``v_max`` cells per second if the cells ahead are free;
* the vehicle on cell 0 can cross the intersection only if its lane has a green
  light, the saturation headway has elapsed and the downstream lane has room;
* a vehicle that does not move during a second is *waiting* (counted in the
  waiting time and in the queue).

After crossing, a vehicle goes straight, turns right or turns left (random) and
enters the incoming lane of the next intersection, or leaves the network if it
was on the border.

The simulator is deliberately simple (pure Python + NumPy) so the project runs
anywhere without installing SUMO, while still exhibiting queues, spill-back,
saturation flow and the effect of the yellow (lost) time.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

# Approach a lane comes FROM. A vehicle on lane "N" comes from the north and
# drives towards the south.
DIRECTIONS: Tuple[str, ...] = ("N", "E", "S", "W")
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
# Heading of vehicles on each approach.
HEADING = {"N": "S", "S": "N", "E": "W", "W": "E"}
# Grid displacement (row, col) when driving in a heading.
MOVE = {"S": (1, 0), "N": (-1, 0), "E": (0, 1), "W": (0, -1)}
RIGHT_OF = {"S": "W", "W": "N", "N": "E", "E": "S"}
LEFT_OF = {"S": "E", "E": "N", "N": "W", "W": "S"}

# Phase 0 gives green to the North-South axis, phase 1 to the East-West axis.
PHASES: Tuple[Tuple[str, ...], ...] = (("N", "S"), ("E", "W"))

CELL_LENGTH = 7.5  # metres occupied by one vehicle (car + gap)

# Simplified CO2 model (orders of magnitude of a petrol passenger car).
CO2_IDLE_G_PER_S = 0.19      # engine idling at a red light
CO2_MOVING_G_PER_S = 1.20    # cruising in the urban network
CO2_RESTART_G = 8.0          # extra emission to re-accelerate after a stop


@dataclass(slots=True)
class Vehicle:
    vid: int
    t_spawn: int
    pos: int
    wait: int = 0
    stops: int = 0
    stopped: bool = False
    last_tick: int = -1


@dataclass
class Lane:
    """Incoming lane of an intersection."""

    inter: Tuple[int, int]
    origin: str
    n_cells: int
    is_boundary: bool
    vehicles: List[Vehicle] = field(default_factory=list)  # sorted by pos
    buffer: Deque[Tuple[int, int]] = field(default_factory=deque)  # (vid, t_spawn) waiting to enter
    next_cross_time: int = 0
    headway_factor: float = 1.0  # > 1 during an accident (capacity drop)
    downstream: Optional["Lane"] = None  # where straight-going vehicles go (None = exit)

    def entry_free(self) -> bool:
        return not self.vehicles or self.vehicles[-1].pos < self.n_cells - 1

    def queue_length(self) -> int:
        return sum(1 for v in self.vehicles if v.stopped) + len(self.buffer)

    def occupancy(self) -> float:
        return len(self.vehicles) / self.n_cells


@dataclass
class SignalState:
    phase: int = 0
    mode: str = "green"  # "green" or "yellow"
    timer: int = 0  # seconds left in the current mode
    next_phase: int = 0
    next_duration: int = 0
    elapsed: int = 0  # seconds since the current green started


DemandFn = Callable[[int, Tuple[int, int], str], float]


class TrafficSimulator:
    """Grid network of signalised intersections."""

    def __init__(
        self,
        rows: int = 1,
        cols: int = 1,
        n_cells: int = 25,
        v_max: int = 2,
        sat_headway: int = 2,
        yellow_time: int = 3,
        turn_probs: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: Optional[int] = None,
    ) -> None:
        self.rows, self.cols = rows, cols
        self.n_cells = n_cells
        self.v_max = v_max
        self.sat_headway = sat_headway
        self.yellow_time = yellow_time
        self.turn_probs = turn_probs  # (straight, right, left)
        self.intersections: List[Tuple[int, int]] = [(r, c) for r in range(rows) for c in range(cols)]
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self._build()

    # ------------------------------------------------------------------ setup
    def _build(self) -> None:
        self.lanes: Dict[Tuple[Tuple[int, int], str], Lane] = {}
        for inter in self.intersections:
            for d in DIRECTIONS:
                self.lanes[(inter, d)] = Lane(inter, d, self.n_cells, self._is_boundary(inter, d))
        for (inter, d), lane in self.lanes.items():
            lane.downstream = self.next_lane(inter, HEADING[d])
        self.signals: Dict[Tuple[int, int], SignalState] = {i: SignalState() for i in self.intersections}
        self.t = 0
        self._next_vid = 0
        # episode statistics
        self.total_wait = 0
        self.total_stops = 0
        self.co2_g = 0.0
        self.spawned = 0
        self.completed = 0
        self.travel_times: List[int] = []
        self.queue_history: List[int] = []

    def _is_boundary(self, inter: Tuple[int, int], origin: str) -> bool:
        """A lane is fed from outside the network if no intersection lies upstream."""
        r, c = inter
        dr, dc = MOVE[origin]  # upstream intersection is in the direction of the origin
        ur, uc = r + dr, c + dc
        return not (0 <= ur < self.rows and 0 <= uc < self.cols)

    def next_lane(self, inter: Tuple[int, int], heading: str) -> Optional[Lane]:
        """Incoming lane reached when leaving ``inter`` towards ``heading``."""
        r, c = inter
        dr, dc = MOVE[heading]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.rows and 0 <= nc < self.cols:
            return self.lanes[((nr, nc), OPPOSITE[heading])]
        return None

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng.seed(seed)
            self.np_rng = np.random.default_rng(seed)
        self._build()

    # ---------------------------------------------------------------- signals
    def is_green(self, lane: Lane) -> bool:
        sig = self.signals[lane.inter]
        return sig.mode == "green" and lane.origin in PHASES[sig.phase]

    def set_signal(self, inter: Tuple[int, int], phase: int, duration: int) -> None:
        """Apply a decision: keep/extend the current phase or switch (with yellow)."""
        sig = self.signals[inter]
        if phase == sig.phase and sig.mode == "green":
            sig.timer = duration
        else:
            sig.mode = "yellow"
            sig.timer = self.yellow_time
            sig.next_phase = phase
            sig.next_duration = duration

    def _update_signals(self) -> List[Tuple[int, int]]:
        """Advance signal timers; return intersections whose green just expired."""
        expired = []
        for inter, sig in self.signals.items():
            sig.timer -= 1
            if sig.mode == "green":
                sig.elapsed += 1
            if sig.timer <= 0:
                if sig.mode == "yellow":
                    sig.mode = "green"
                    sig.phase = sig.next_phase
                    sig.timer = sig.next_duration
                    sig.elapsed = 0
                else:
                    expired.append(inter)
        return expired

    # ---------------------------------------------------------------- dynamics
    def _choose_exit(self, lane: Lane) -> Optional[Lane]:
        heading = HEADING[lane.origin]
        u = self.rng.random()
        s, r, _ = self.turn_probs
        if u < s:
            new_heading = heading
        elif u < s + r:
            new_heading = RIGHT_OF[heading]
        else:
            new_heading = LEFT_OF[heading]
        return self.next_lane(lane.inter, new_heading)

    def _move_lane(self, lane: Lane) -> None:
        t = self.t
        green = self.is_green(lane)
        front_limit = 0  # first cell the next vehicle may occupy
        kept: List[Vehicle] = []
        for v in lane.vehicles:
            moved = 0
            if v.last_tick == t:  # already moved this tick (just entered from upstream)
                kept.append(v)
                front_limit = v.pos + 1
                continue
            v.last_tick = t
            if v.pos == 0 and not kept:
                headway = int(round(self.sat_headway * lane.headway_factor))
                if green and t >= lane.next_cross_time:
                    target = self._choose_exit(lane)
                    if target is None or target.entry_free():
                        lane.next_cross_time = t + headway
                        self._account(v, moved=True)
                        if target is None:
                            self.completed += 1
                            self.travel_times.append(t - v.t_spawn)
                        else:
                            v.pos = target.n_cells - 1
                            target.vehicles.append(v)
                        continue
            else:
                new_pos = max(v.pos - self.v_max, front_limit)
                if new_pos < v.pos:
                    moved = v.pos - new_pos
                    v.pos = new_pos
            self._account(v, moved=moved > 0)
            kept.append(v)
            front_limit = v.pos + 1
        lane.vehicles = kept

    def _account(self, v: Vehicle, moved: bool) -> None:
        if moved:
            if v.stopped:
                self.co2_g += CO2_RESTART_G
            v.stopped = False
            self.co2_g += CO2_MOVING_G_PER_S
        else:
            if not v.stopped:
                v.stops += 1
                self.total_stops += 1
            v.stopped = True
            v.wait += 1
            self.total_wait += 1
            self.co2_g += CO2_IDLE_G_PER_S

    def _spawn(self, demand: DemandFn) -> None:
        for lane in self.lanes.values():
            if not lane.is_boundary:
                continue
            rate = demand(self.t, lane.inter, lane.origin)
            if rate > 0 and self.rng.random() < rate:
                lane.buffer.append((self._next_vid, self.t))
                self._next_vid += 1
                self.spawned += 1
            if lane.buffer and lane.entry_free():
                vid, t_spawn = lane.buffer.popleft()
                waited = self.t - t_spawn
                v = Vehicle(vid=vid, t_spawn=t_spawn, pos=lane.n_cells - 1, wait=waited, last_tick=self.t)
                lane.vehicles.append(v)
            # vehicles still blocked outside the network are waiting too
            self.total_wait += len(lane.buffer)
            self.co2_g += CO2_IDLE_G_PER_S * len(lane.buffer)

    def step(self, demand: DemandFn) -> List[Tuple[int, int]]:
        """Simulate one second. Returns intersections that need a new decision."""
        expired = self._update_signals()
        # Process lanes closest to the exit first is not required: entry checks
        # use the live occupancy of the downstream lane.
        for lane in self.lanes.values():
            self._move_lane(lane)
        self._spawn(demand)
        self.queue_history.append(self.total_queue())
        self.t += 1
        return expired

    # ------------------------------------------------------------- observables
    def total_queue(self) -> int:
        return sum(l.queue_length() for l in self.lanes.values())

    def intersection_queue(self, inter: Tuple[int, int]) -> int:
        return sum(self.lanes[(inter, d)].queue_length() for d in DIRECTIONS)

    def intersection_waiting(self, inter: Tuple[int, int]) -> int:
        """Vehicles currently stopped on the incoming lanes of ``inter``."""
        return self.intersection_queue(inter)

    def lane_mean_wait(self, lane: Lane) -> float:
        waits = [v.wait for v in lane.vehicles] + [self.t - ts for _, ts in lane.buffer]
        return float(np.mean(waits)) if waits else 0.0

    def vehicles_in_network(self) -> int:
        return sum(len(l.vehicles) + len(l.buffer) for l in self.lanes.values())

    def metrics(self) -> Dict[str, float]:
        n = max(self.spawned, 1)
        return {
            "mean_waiting_time": self.total_wait / n,
            "mean_travel_time": float(np.mean(self.travel_times)) if self.travel_times else 0.0,
            "mean_queue_length": float(np.mean(self.queue_history)) if self.queue_history else 0.0,
            "throughput": float(self.completed),
            "stops_per_vehicle": self.total_stops / n,
            "co2_kg": self.co2_g / 1000.0,
            "spawned": float(self.spawned),
        }
