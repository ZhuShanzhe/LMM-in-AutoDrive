"""Competition-oriented basic voice route with stable city traffic."""

from __future__ import annotations

import os

from scenarios.basic.urban_traffic import FixedRouteTraffic
from scenarios.basic.voice_control_5km import BasicVoiceControl5KmScenario


class UrbanVoice5KmScenario(BasicVoiceControl5KmScenario):
    """5 km, three-lane-per-direction city route with 15 parsed commands."""

    default_map = "Town04_Opt"
    default_duration_s = 480.0

    def __init__(self, world, external_control=True, config_path=None):
        config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "configs", "basic_voice_urban_5km.json",
        )
        super().__init__(world, external_control=external_control, config_path=config_path)
        self.scenario_id = self.config["scenario_id"]
        self.scenario_name = self.config["scenario_name"]
        self.traffic = None
        self.lifecycle_log = []

    def setup(self):
        super().setup()
        self.traffic = FixedRouteTraffic(
            self.world, self.client, self.route_manager, self.config.get("traffic", {}),
        )
        for actor in self.traffic.setup():
            self.add_actor(actor, "traffic_{0}".format(actor.id))
        self.lifecycle_log.append({
            "type": "fixed_route_traffic_initialized",
            "traffic": self.traffic.snapshot(),
        })

    def tick(self):
        super().tick()
        if self.traffic is not None:
            self.traffic.tick(self.ego_vehicle, self.route_manager.progress_m)

    def destroy(self):
        if self.traffic is not None:
            self.traffic.destroy()
        super().destroy()

    def report_events(self, events):
        """Ignore only the generated-map chassis-settle contact at startup."""
        progress = float(self.route_manager.progress_m)
        exempt_windows = self.config.get("map_lane_marking_exemptions_m", [])
        if any(len(window) == 2 and float(window[0]) <= progress <= float(window[1]) for window in exempt_windows):
            filtered = [
                item for item in events.get("new_lane_invasion_events", [])
                if not self._is_illegal_lane_invasion(item)
            ]
            if len(filtered) != len(events.get("new_lane_invasion_events", [])):
                events = dict(events)
                events["new_lane_invasion_events"] = filtered
                events["lane_invasion"] = bool(filtered)
                self.lifecycle_log.append({
                    "type": "map_lane_marking_transition_ignored",
                    "route_progress_m": round(progress, 1),
                })
        accepted = [
            item for item in events.get("new_collision_events", [])
            if item.get("other_actor_type") != "static.unknown"
        ]
        if len(accepted) != len(events.get("new_collision_events", [])):
            events = dict(events)
            events["new_collision_events"] = accepted
            events["collision"] = bool(accepted)
            events["collision_count"] = int(self.metrics.get("collision_count", 0)) + len(accepted)
            self.lifecycle_log.append({
                "type": "generated_map_road_contact_ignored",
                "simulation_time_s": round(float(self.metrics["simulation_time"]), 3),
            })
        super().report_events(events)

    def get_status(self):
        status = super().get_status()
        status["traffic"] = self.traffic.snapshot() if self.traffic else {}
        return status

    def get_scenario_info(self):
        info = super().get_scenario_info()
        info["traffic"] = self.config.get("traffic", {})
        info["traffic_design"] = (
            "fixed Traffic Manager actor pool with complete preassigned routes"
        )
        return info

    def drain_event_log(self):
        result = list(self.lifecycle_log)
        self.lifecycle_log = []
        return result
