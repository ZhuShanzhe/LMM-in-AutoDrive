"""Schedule independently owned event actors along one ego route."""

import json

from continuous.route_manager import RouteManager


class ScenarioManager:
    def __init__(self, world, route_manager=None):
        self.world = world
        self.route_manager = route_manager or RouteManager(world)
        self.factories = {}
        self.events = []
        self.active = []
        self.event_log = []

    def register(self, name, factory):
        self.factories[name] = factory

    def load(self, config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        route = config["route"]
        self.route_manager.build_route(
            length_m=route["length_m"],
            step_m=route.get("step_m", 5.0),
        )
        self.set_events(config["events"])

    def set_events(self, events):
        self.events = []
        self.event_log = []
        for event in events:
            normalized = dict(event)
            normalized["triggered"] = False
            normalized["status"] = "PENDING"
            self.events.append(normalized)

    def tick(self, ego_vehicle):
        progress_m = self.route_manager.update(ego_vehicle)
        active_before_tick = list(self.active)
        for event in self.events:
            if not event["triggered"] and progress_m >= float(event["distance_m"]):
                self._activate(event, ego_vehicle, progress_m)
                event["triggered"] = True
        for active in active_before_tick:
            active.tick()
            if active.finished():
                source = getattr(active, "event", {})
                source["status"] = getattr(active, "status", "COMPLETED")
                transition = "failed" if source["status"] == "FAILED" else "completed"
                self._log(transition, source, progress_m, self._status_of(active))
                active.destroy()
                self.active.remove(active)
        return progress_m

    def destroy(self):
        for active in self.active:
            active.destroy()
        self.active = []

    def snapshot(self):
        active_details = []
        for active in self.active:
            details = dict(self._status_of(active))
            details["id"] = getattr(active, "event", {}).get("id")
            details["scenario"] = getattr(active, "event", {}).get("scenario")
            active_details.append(details)
        return {
            "pending": [event["id"] for event in self.events if event["status"] == "PENDING"],
            "active": [event["id"] for event in self.events if event["status"] == "ACTIVE"],
            "active_details": active_details,
            "completed": [event["id"] for event in self.events if event["status"] == "COMPLETED"],
            "failed": [event["id"] for event in self.events if event["status"] == "FAILED"],
            "events": [dict(event) for event in self.events],
        }

    def drain_event_log(self):
        result = self.event_log
        self.event_log = []
        return result

    def _activate(self, event, ego_vehicle, progress_m):
        name = event["scenario"]
        try:
            factory = self.factories[name]
        except KeyError:
            raise KeyError("No event factory registered for {0}".format(name))
        scenario = factory(self.world, ego_vehicle, event)
        scenario.event = event
        scenario.setup()
        event["status"] = "ACTIVE"
        self._log("activated", event, progress_m, self._status_of(scenario))
        self.active.append(scenario)

    def _log(self, transition, event, progress_m, details):
        self.event_log.append({
            "type": "scenario_event",
            "transition": transition,
            "event_id": event.get("id"),
            "event_type": event.get("scenario"),
            "route_progress_m": round(float(progress_m), 3),
            "details": details,
        })

    @staticmethod
    def _status_of(scenario):
        getter = getattr(scenario, "get_status", None)
        if getter is None:
            return {"status": getattr(scenario, "status", "COMPLETED")}
        return getter()
