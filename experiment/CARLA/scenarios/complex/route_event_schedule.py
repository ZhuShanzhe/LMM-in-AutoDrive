"""Deterministic route-progress event state for complex driving scenarios."""

from __future__ import annotations


class RouteEventSchedule:
    """Activate configured scenario events once as the ego reaches each marker.

    Actor creation remains scenario-owned.  Keeping this class pure makes event
    order, logs, and recovery behavior testable without a CARLA server.
    """

    def __init__(self, events):
        self.events = sorted((dict(event) for event in events or []), key=self._distance)
        self._states = {
            event["id"]: {"status": "PENDING", "activated_at_m": None, "reason": None}
            for event in self.events
        }
        self._history = []

    def update(self, progress_m):
        """Return newly activated events at the supplied route progress."""
        progress_m = float(progress_m)
        activated = []
        for event in self.events:
            event_id = event["id"]
            state = self._states[event_id]
            if state["status"] != "PENDING" or progress_m < self._distance(event):
                continue
            state.update({"status": "ACTIVE", "activated_at_m": round(progress_m, 3)})
            record = {
                "id": event_id,
                "status": "ACTIVE",
                "route_progress_m": round(progress_m, 3),
                "actors": list(event.get("actors", [])),
                "behavior": event.get("behavior", ""),
            }
            self._history.append(record)
            activated.append(dict(event))
        return activated

    def complete(self, event_id, reason):
        state = self._states.get(str(event_id))
        if state is None or state["status"] != "ACTIVE":
            return False
        state.update({"status": "COMPLETED", "reason": str(reason)})
        self._history.append({"id": str(event_id), "status": "COMPLETED", "reason": str(reason)})
        return True

    def snapshot(self):
        return {
            "event_count": len(self.events),
            "states": {event_id: dict(state) for event_id, state in self._states.items()},
            "history": list(self._history),
        }

    @staticmethod
    def _distance(event):
        if "at_m" not in event:
            raise ValueError("complex event requires at_m")
        if not event.get("id"):
            raise ValueError("complex event requires id")
        return float(event["at_m"])
