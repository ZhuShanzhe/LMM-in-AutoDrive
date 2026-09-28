"""State shared by ego-following background-vehicle sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from typing import Any


@dataclass
class LaneTrafficSource:
    waypoint: Any
    route_s_m: float
    actors: list[Any] = field(default_factory=list)
    last_spawn_progress_m: float = -inf
