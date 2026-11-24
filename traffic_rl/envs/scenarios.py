"""Traffic demand scenarios.

A scenario gives, for every second ``t`` and every boundary lane, the
probability that a new vehicle arrives (Bernoulli approximation of a Poisson
process, in vehicles / second). It can also schedule *incidents* that reduce the
capacity of a lane (an accident near the stop line multiplies the saturation
headway, so fewer vehicles can cross per green second).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

Inter = Tuple[int, int]


@dataclass
class Incident:
    inter: Inter
    origin: str
    t_start: int
    t_end: int
    headway_factor: float = 3.0


@dataclass
class Scenario:
    name: str
    description: str
    rate_fn: Callable[[int, Inter, str, int], float]
    incidents: List[Incident] = field(default_factory=list)

    def demand(self, episode_length: int) -> Callable[[int, Inter, str], float]:
        return lambda t, inter, origin: self.rate_fn(t, inter, origin, episode_length)


def _bump(t: int, center: float, width: float) -> float:
    return math.exp(-0.5 * ((t - center) / width) ** 2)


def _normal(t, inter, origin, T):
    return 0.14


def _low(t, inter, origin, T):
    return 0.05


def _unbalanced(t, inter, origin, T):
    return 0.22 if origin in ("N", "S") else 0.06


def _rush_hour(t, inter, origin, T):
    # Morning peak: a strong commuter flow on the North-South axis.
    peak = _bump(t, T * 0.5, T / 6)
    if origin in ("N", "S"):
        return 0.08 + 0.18 * peak
    return 0.06 + 0.04 * peak


def _accident(t, inter, origin, T):
    return 0.14


def _event(t, inter, origin, T):
    # A sports event ends: a short, very intense surge coming from the West.
    surge = _bump(t, T * 0.4, T / 12)
    if origin == "W":
        return 0.08 + 0.26 * surge
    return 0.09


def build_scenarios(rows: int = 1, cols: int = 1, episode_length: int = 3600) -> Dict[str, Scenario]:
    """All scenarios available for a ``rows x cols`` grid."""
    T = episode_length
    first = (0, 0)
    return {
        "low": Scenario("low", "Light traffic at night (0.05 veh/s per entry).", _low),
        "normal": Scenario("normal", "Balanced daytime traffic (0.14 veh/s per entry).", _normal),
        "unbalanced": Scenario("unbalanced", "Heavy North-South flow, light East-West flow.", _unbalanced),
        "rush_hour": Scenario(
            "rush_hour", "Gaussian morning peak on the North-South axis (up to 0.26 veh/s).", _rush_hour
        ),
        "accident": Scenario(
            "accident",
            "Accident on the North approach of the first intersection between T/3 and 2T/3 "
            "(capacity divided by 3).",
            _accident,
            incidents=[Incident(first, "N", T // 3, 2 * T // 3, 3.0)],
        ),
        "event": Scenario("event", "End of a sports event: short surge coming from the West.", _event),
    }


TRAIN_SCENARIOS = ("normal", "unbalanced", "rush_hour", "accident")
ALL_SCENARIOS = ("low", "normal", "unbalanced", "rush_hour", "accident", "event")
