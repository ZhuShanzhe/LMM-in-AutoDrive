"""Decision-provider boundaries for the CARLA experiment runner.

The JSON-file provider is a temporary process boundary: a parser, risk module,
or planner can atomically replace one JSON document without importing CARLA or
sharing Python-version-specific runtime code with the simulator process.
"""

import json
import os

from control.protocol import normalize_intent


class JsonFileDecisionPolicy:
    """Read a current DrivingIntent or ControlDecision document from disk.

    A malformed or unavailable external decision must never leave the ego
    vehicle executing a stale command, so the provider emits a safe stop.
    """

    def __init__(self, path, default_speed_kmh=25.0):
        self.path = os.path.abspath(path)
        self.default_speed_kmh = float(default_speed_kmh)

    def decide(self, world_state):
        del world_state
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                document = json.load(handle)
            return normalize_intent(document, self.default_speed_kmh)
        except (IOError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return {
                "action": "stop",
                "target_speed_kmh": 0.0,
                "emergency": False,
                "reason": "external_decision_unavailable_{0}".format(
                    type(exc).__name__.lower()
                ),
            }
