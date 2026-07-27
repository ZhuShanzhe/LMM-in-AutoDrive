"""Focused decision/control validation with a braking lead vehicle and traffic flow."""

from __future__ import annotations

import carla

from scenarios.emergency.emergency_brake import EmergencyBrakeScenario


class BrakingWithTrafficValidationScenario(EmergencyBrakeScenario):
    """Exercise safe following while non-triggering traffic remains visible.

    The lead vehicle is the only scripted hazard. Background vehicles are
    intentionally placed outside its initial safety gap and remain under CARLA
    Traffic Manager so they provide traffic flow without becoming hidden test
    triggers.
    """

    scenario_id = "braking_with_traffic_validation"

    def __init__(self, world, external_control=True):
        super().__init__(world, external_control)
        self.background_traffic = []
        self.background_offsets_m = (42.0, 68.0, 96.0, 124.0)
        self.background_speed_kmh = 42.0

    def setup(self):
        # Start outside the medium-distance threshold so the run demonstrates
        # a true low-risk cruise followed by a hazard-driven transition.
        self.initial_distance = 35.0
        self.brake_trigger_frame = 120
        super().setup()
        self.scenario_id = "braking_with_traffic_validation"
        self.scenario_name = "Lead vehicle braking with background traffic"
        self._spawn_background_traffic()

    def _spawn_background_traffic(self):
        ego_location = self.ego_vehicle.get_location()
        world_map = self.world.get_map()
        ego_waypoint = world_map.get_waypoint(
            ego_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if ego_waypoint is None:
            return
        blueprint_library = self.world.get_blueprint_library()
        vehicle_types = (
            "vehicle.audi.tt",
            "vehicle.lincoln.mkz_2020",
            "vehicle.tesla.model3",
            "vehicle.volkswagen.t2",
        )
        for index, offset_m in enumerate(self.background_offsets_m):
            candidates = ego_waypoint.next(offset_m)
            if not candidates:
                continue
            waypoint = self._adjacent_driving_lane(candidates[0], index)
            if waypoint is None or self._is_occupied(waypoint.transform.location):
                continue
            blueprint = blueprint_library.find(vehicle_types[index % len(vehicle_types)])
            if blueprint.has_attribute("role_name"):
                blueprint.set_attribute("role_name", "background_traffic")
            transform = carla.Transform(
                carla.Location(
                    x=waypoint.transform.location.x,
                    y=waypoint.transform.location.y,
                    z=waypoint.transform.location.z + 0.25,
                ),
                waypoint.transform.rotation,
            )
            actor = self.world.try_spawn_actor(blueprint, transform)
            if actor is None:
                continue
            # CARLA's default Traffic Manager owns background traffic. The
            # scenario keeps the lead vehicle under explicit scripted control.
            actor.set_autopilot(True)
            self.background_traffic.append(actor)
            self.add_actor(actor, "background_traffic_{0}".format(index))

    @staticmethod
    def _adjacent_driving_lane(waypoint, index):
        primary = waypoint.get_left_lane() if index % 2 == 0 else waypoint.get_right_lane()
        secondary = waypoint.get_right_lane() if index % 2 == 0 else waypoint.get_left_lane()
        for candidate in (primary, secondary, waypoint):
            if candidate is not None and candidate.lane_type == carla.LaneType.Driving:
                return candidate
        return None

    def _is_occupied(self, location):
        for actor in self.world.get_actors().filter("vehicle.*"):
            if actor.is_alive and actor.get_location().distance(location) < 10.0:
                return True
        return False

    def get_status(self):
        status = super().get_status()
        status["traffic"] = {
            "background_actor_count": sum(
                1 for actor in self.background_traffic if actor is not None and actor.is_alive
            ),
            "background_actor_ids": [
                actor.id for actor in self.background_traffic if actor is not None and actor.is_alive
            ],
        }
        return status
