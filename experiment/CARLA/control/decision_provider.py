"""Decision-provider boundaries for the CARLA experiment runner.

The JSON-file provider is a temporary process boundary: a parser, risk module,
or planner can atomically replace one JSON document without importing CARLA or
sharing Python-version-specific runtime code with the simulator process.
"""

import json
import os
import re

from control.protocol import normalize_intent


class JsonFileDecisionPolicy:
    """Read a current DrivingIntent or ControlDecision document from disk.

    A malformed or unavailable external decision must never leave the ego
    vehicle executing a stale command, so the provider emits a safe stop.
    """

    def __init__(self, path, default_speed_kmh=25.0, max_age_frames=None):
        self.path = os.path.abspath(path)
        self.default_speed_kmh = float(default_speed_kmh)
        if max_age_frames is not None and int(max_age_frames) < 0:
            raise ValueError("max_age_frames must be non-negative or None")
        self.max_age_frames = (
            None if max_age_frames is None else int(max_age_frames)
        )
        self._last_telemetry = {
            "source": "json_file",
            "decision_path": self.path,
            "max_age_frames": self.max_age_frames,
            "status": "not_read",
        }

    def decide(self, world_state):
        try:
            with open(self.path, "r", encoding="utf-8-sig") as handle:
                document = json.load(handle)
            decision = normalize_intent(document, self.default_speed_kmh)
            return self._validate_frame(decision, document, world_state)
        except (IOError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return self._safe_stop(
                "external_decision_unavailable_{0}".format(
                    type(exc).__name__.lower()
                )
            )

    def telemetry(self):
        return dict(self._last_telemetry)

    @staticmethod
    def _frame_number(value):
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, int):
            return value
        match = re.search(r"(\d+)$", str(value).strip())
        return int(match.group(1)) if match else None

    def _safe_stop(self, reason, **details):
        self._last_telemetry = {
            "source": "json_file",
            "decision_path": self.path,
            "max_age_frames": self.max_age_frames,
            "status": "safe_stop",
            "reason": reason,
            **details,
        }
        return {
            "action": "stop",
            "target_speed_kmh": 0.0,
            "emergency": False,
            "reason": reason,
        }

    def _validate_frame(self, decision, document, world_state):
        decision_frame_id = document.get("frame_id")
        telemetry = {
            "source": "json_file",
            "decision_path": self.path,
            "max_age_frames": self.max_age_frames,
            "decision_frame_id": decision_frame_id,
        }
        if self.max_age_frames is None:
            telemetry["status"] = "accepted_without_frame_guard"
            self._last_telemetry = telemetry
            return decision

        if not isinstance(world_state, dict):
            return self._safe_stop("external_decision_missing_current_frame")
        current_frame = self._frame_number(
            world_state.get("simulation_frame", world_state.get("frame_id"))
        )
        decision_frame = self._frame_number(decision_frame_id)
        if current_frame is None:
            return self._safe_stop("external_decision_missing_current_frame")
        if decision_frame is None:
            return self._safe_stop(
                "external_decision_missing_frame",
                current_frame=current_frame,
            )

        decision_age_frames = current_frame - decision_frame
        if decision_age_frames < 0:
            return self._safe_stop(
                "external_decision_future_frame",
                current_frame=current_frame,
                decision_frame=decision_frame,
            )
        if decision_age_frames > self.max_age_frames:
            return self._safe_stop(
                "external_decision_stale",
                current_frame=current_frame,
                decision_frame=decision_frame,
                decision_age_frames=decision_age_frames,
            )

        decision["decision_frame_id"] = decision_frame_id
        decision["decision_age_frames"] = decision_age_frames
        telemetry.update({
            "status": "accepted",
            "current_frame": current_frame,
            "decision_age_frames": decision_age_frames,
        })
        self._last_telemetry = telemetry
        return decision
