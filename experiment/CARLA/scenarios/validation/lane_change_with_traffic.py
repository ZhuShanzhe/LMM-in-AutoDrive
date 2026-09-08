"""Short, repeatable ego lane-change validation with visible traffic flow."""

from __future__ import annotations

import carla

from scenarios.base import BaseScenario
from scenarios.utils.road import RoadFinder
from scenarios.utils.vehicle import VehicleSpawner


class LaneChangeWithTrafficValidationScenario(BaseScenario):
    default_map = "Town04"

    def __init__(self, world, external_control=True):
        super().__init__(world, external_control)
        self.vehicle_spawner = VehicleSpawner(world)
        self.road_finder = RoadFinder(world)
        self.ego_vehicle = None
        self.initial_lane_id = None
        self.target_lane_id = None
        self.background_traffic = []
        self.collision_sensor = None
        self._target_lane_hold_s = 0.0
        self.timeout_s = 30.0
        self.success_condition = {
            "type": "safe_lane_change_and_stabilize",
            "hold_s": 1.0,
            "background_traffic_required": True,
        }
        self.failure_conditions = ["collision", "timeout"]

    def setup(self):
        road = self.road_finder.find_straight_road(min_length=180)
        base = road["waypoint"]
        candidates = base.previous(50.0)
        if not candidates:
            raise RuntimeError("Cannot locate ego validation waypoint")
        ego_waypoint = candidates[0]
        target = ego_waypoint.get_left_lane()
        if target is None or target.lane_type != carla.LaneType.Driving:
            target = ego_waypoint.get_right_lane()
        if target is None or target.lane_type != carla.LaneType.Driving:
            raise RuntimeError("Validation road has no adjacent driving lane")
        self.initial_lane_id = ego_waypoint.lane_id
        self.target_lane_id = target.lane_id
        transform = carla.Transform(
            carla.Location(
                x=ego_waypoint.transform.location.x,
                y=ego_waypoint.transform.location.y,
                z=ego_waypoint.transform.location.z + 0.3,
            ),
            ego_waypoint.transform.rotation,
        )
        self.ego_vehicle = self.vehicle_spawner.spawn_ego_vehicle(transform)
        if self.ego_vehicle is None:
            raise RuntimeError("Ego spawn failed")
        self.add_actor(self.ego_vehicle, "ego")
        self._spawn_background_traffic(ego_waypoint)
        sensor_bp = self.world.get_blueprint_library().find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            sensor_bp, carla.Transform(), attach_to=self.ego_vehicle
        )
        self.collision_sensor.listen(self._on_collision)
        self.add_actor(self.collision_sensor, "collision_sensor")
        self.status = "RUNNING"
        self.scenario_id = "lane_change_with_traffic_validation"
        self.scenario_name = "Safe ego lane change with background traffic"

    def _spawn_background_traffic(self, ego_waypoint):
        library = self.world.get_blueprint_library()
        for index, offset_m in enumerate((45.0, 75.0, 110.0, 145.0)):
            next_waypoints = ego_waypoint.next(offset_m)
            if not next_waypoints:
                continue
            waypoint = next_waypoints[0]
            if index % 2:
                adjacent = waypoint.get_right_lane()
                if adjacent is not None and adjacent.lane_type == carla.LaneType.Driving:
                    waypoint = adjacent
            blueprint = library.find(("vehicle.audi.tt", "vehicle.tesla.model3")[index % 2])
            transform = carla.Transform(
                carla.Location(x=waypoint.transform.location.x, y=waypoint.transform.location.y, z=waypoint.transform.location.z + 0.25),
                waypoint.transform.rotation,
            )
            actor = self.world.try_spawn_actor(blueprint, transform)
            if actor is not None:
                actor.set_autopilot(True)
                self.background_traffic.append(actor)
                self.add_actor(actor, "background_traffic_{0}".format(index))

    def tick(self):
        if self._finished:
            return
        self.metrics["simulation_time"] += self.fixed_delta_s
        if self.metrics["simulation_time"] > self.timeout_s:
            self.failure("timeout")
            return
        waypoint = self.world.get_map().get_waypoint(
            self.ego_vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if waypoint is not None and waypoint.lane_id == self.target_lane_id:
            self._target_lane_hold_s += self.fixed_delta_s
            if self._target_lane_hold_s >= 1.0:
                self.success("lane_change_completed_and_stable")
        else:
            self._target_lane_hold_s = 0.0

    def _on_collision(self, event):
        self.metrics["collision_count"] += 1
        self.failure("collision_with_{0}".format(event.other_actor.type_id))

    def get_status(self):
        return {
            "status": self.status,
            "reason": self.reason,
            "actors": self.get_actor_ids(),
            "metrics": self.metrics,
            "initial_lane_id": self.initial_lane_id,
            "target_lane_id": self.target_lane_id,
            "traffic": {"background_actor_count": sum(1 for actor in self.background_traffic if actor.is_alive)},
        }

    def get_ego_vehicle(self):
        return self.ego_vehicle

    def get_policy_context(self):
        return {"default_speed_kmh": 35.0}
