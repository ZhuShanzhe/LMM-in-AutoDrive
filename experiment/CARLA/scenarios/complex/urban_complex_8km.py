"""Scene 2 runtime scaffold: long route, mixed traffic, and scripted events."""

from __future__ import annotations

import os

from scenarios.basic.urban_traffic import EgoCentricTraffic
from scenarios.basic.voice_control_5km import BasicVoiceControl5KmScenario
from scenarios.complex.route_event_schedule import RouteEventSchedule


class UrbanComplex8KmScenario(BasicVoiceControl5KmScenario):
    """Configurable Scene 2 runner with auditable event activation.

    The class deliberately relies on route directives for turns and U-turns.
    A map missing those real connections fails the inherited route preflight
    instead of silently replacing a required manoeuvre with straight driving.
    """

    default_map = "Town10HD_Opt"
    default_duration_s = 780.0

    def __init__(self, world, external_control=True, config_path=None):
        config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "configs", "complex_avoidance_8km_runtime.json",
        )
        super().__init__(world, external_control=external_control, config_path=config_path)
        self.scenario_id = self.config["scenario_id"]
        self.scenario_name = self.config["scenario_name"]
        self.traffic = None
        self.event_schedule = RouteEventSchedule(self.config.get("special_events", []))
        self.lifecycle_log = []

    def setup(self):
        super().setup()
        self.traffic = EgoCentricTraffic(
            self.world, self.client, self.route_manager, self.config.get("traffic", {}),
        )
        for actor in self.traffic.setup(self.ego_vehicle):
            self.add_actor(actor, "mixed_traffic_{0}".format(actor.id))
        self.lifecycle_log.append({
            "type": "mixed_traffic_initialized",
            "traffic": self.traffic.snapshot(),
        })

    def tick(self):
        super().tick()
        if self.traffic is not None:
            self.traffic.tick(self.ego_vehicle, self.route_manager.progress_m)
        for event in self.event_schedule.update(self.route_manager.progress_m):
            # The event is intentionally logged before its actor adapter is
            # attached, preserving a deterministic trigger/audit contract.
            self.lifecycle_log.append({
                "type": "complex_event_activated",
                "event_id": event["id"],
                "route_progress_m": round(self.route_manager.progress_m, 3),
                "actors": list(event.get("actors", [])),
            })

    def destroy(self):
        if self.traffic is not None:
            self.traffic.destroy()
        super().destroy()

    def get_status(self):
        status = super().get_status()
        status["traffic"] = self.traffic.snapshot() if self.traffic else {}
        status["scenario_events"] = self.event_schedule.snapshot()
        return status

    def get_scenario_info(self):
        info = super().get_scenario_info()
        info["complex_scene_contract"] = self.config.get("runtime_contract", {})
        return info

    def drain_event_log(self):
        result = list(self.lifecycle_log)
        self.lifecycle_log = []
        return result
