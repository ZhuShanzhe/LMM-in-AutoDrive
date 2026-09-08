import fnmatch
import unittest

from scene_understanding.core.carla_world_state import CarlaWorldStateCollector, classify_lane_relation
from scene_understanding.core.world_state import validate_world_state


class FakeVector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class FakeRotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch, self.yaw, self.roll = pitch, yaw, roll


class FakeTransform:
    def __init__(self, location, rotation=None):
        self.location = location
        self.rotation = rotation or FakeRotation()


class FakeWaypoint:
    def __init__(self, road_id, lane_id, *, left=None, right=None, is_junction=False):
        self.road_id = road_id
        self.section_id = 0
        self.lane_id = lane_id
        self.lane_type = "LaneType.Driving"
        self.lane_change = "LaneChange.Both"
        self.is_junction = is_junction
        self._left = left
        self._right = right

    def get_left_lane(self):
        return self._left

    def get_right_lane(self):
        return self._right


class FakeMap:
    def __init__(self, waypoints):
        self.waypoints = waypoints

    def get_waypoint(self, location, project_to_road=True):
        return self.waypoints.get((location.x, location.y))


class FakeActor:
    def __init__(self, actor_id, type_id, x, y, *, velocity=None, state=None):
        self.id = actor_id
        self.type_id = type_id
        self.is_alive = True
        self._transform = FakeTransform(FakeVector(x, y, 0.0))
        self._velocity = velocity or FakeVector()
        self._state = state

    def get_transform(self):
        return self._transform

    def get_location(self):
        return self._transform.location

    def get_velocity(self):
        return self._velocity

    def get_acceleration(self):
        return FakeVector()

    def get_state(self):
        return self._state


class FakeActorList(list):
    def filter(self, pattern):
        return FakeActorList(
            actor for actor in self if fnmatch.fnmatch(actor.type_id, pattern)
        )


class FakeTimestamp:
    elapsed_seconds = 6.15


class FakeSnapshot:
    frame = 123
    timestamp = FakeTimestamp()


class FakeWeather:
    precipitation = 50.0
    fog_density = 0.0


class FakeWorld:
    def __init__(self, actors, carla_map):
        self.actors = FakeActorList(actors)
        self.carla_map = carla_map

    def get_snapshot(self):
        return FakeSnapshot()

    def get_actors(self):
        return self.actors

    def get_map(self):
        return self.carla_map

    def get_weather(self):
        return FakeWeather()


class CarlaWorldStateTests(unittest.TestCase):
    def setUp(self):
        left_lane = FakeWaypoint(12, -2)
        ego_lane = FakeWaypoint(12, -1, left=left_lane)
        ego = FakeActor(1, "vehicle.tesla.model3", 0.0, 0.0, velocity=FakeVector(10, 0, 0))
        front = FakeActor(2, "vehicle.audi.tt", 20.0, 0.0, velocity=FakeVector(6, 0, 0))
        left = FakeActor(3, "vehicle.lincoln.mkz_2020", 10.0, 4.0, velocity=FakeVector(8, 0, 0))
        light = FakeActor(4, "traffic.traffic_light", 30.0, 0.0, state="TrafficLightState.Red")
        far_vehicle = FakeActor(5, "vehicle.dodge.charger", 200.0, 0.0)
        walker = FakeActor(
            6,
            "walker.pedestrian.0001",
            15.0,
            4.0,
            velocity=FakeVector(0, -1, 0),
        )
        cone = FakeActor(7, "static.prop.trafficcone01", 12.0, -3.0)
        barrier = FakeActor(8, "static.prop.streetbarrier", 18.0, -3.0)
        carla_map = FakeMap(
            {
                (0.0, 0.0): ego_lane,
                (20.0, 0.0): ego_lane,
                (10.0, 4.0): left_lane,
            }
        )
        self.ego = ego
        self.world = FakeWorld(
            [ego, front, left, light, far_vehicle, walker, cone, barrier],
            carla_map,
        )

    def test_collects_schema_valid_snapshot(self):
        state = CarlaWorldStateCollector(self.world, self.ego).collect()
        self.assertEqual(validate_world_state(state), [])
        self.assertEqual(state["simulation_frame"], 123)
        self.assertEqual(state["environment"]["weather"], "rain")
        self.assertEqual(state["ego"]["adjacent_lanes"]["left"]["lane_id"], -2)

    def test_collects_metric_relative_motion(self):
        state = CarlaWorldStateCollector(self.world, self.ego).collect()
        front = next(item for item in state["objects"] if item["source_object_id"] == "2")
        self.assertAlmostEqual(front["distance_m"], 20.0)
        self.assertAlmostEqual(front["relative_longitudinal_speed_mps"], -4.0)
        self.assertAlmostEqual(front["closing_speed_mps"], 4.0)
        self.assertEqual(front["lane_relation"], "ego_lane")

    def test_filters_objects_beyond_radius(self):
        state = CarlaWorldStateCollector(self.world, self.ego, max_distance_m=80).collect()
        self.assertNotIn("5", {item["source_object_id"] for item in state["objects"]})

    def test_skips_nonfinite_non_ego_actor_without_dropping_frame(self):
        invalid = FakeActor(99, "vehicle.audi.tt", float("nan"), 0.0)
        self.world.actors.append(invalid)
        collector = CarlaWorldStateCollector(self.world, self.ego)

        state = collector.collect()

        self.assertEqual(validate_world_state(state), [])
        self.assertNotIn(
            "99", {item["source_object_id"] for item in state["objects"]}
        )
        self.assertEqual(collector.last_dropped_actor_count, 1)
        self.assertEqual(collector.total_dropped_actor_samples, 1)

    def test_collects_traffic_light_state(self):
        state = CarlaWorldStateCollector(self.world, self.ego).collect()
        light = next(item for item in state["objects"] if item["category"] == "traffic_light")
        self.assertEqual(light["traffic_light_state"], "red")

    def test_classifies_safety_relevant_static_props(self):
        state = CarlaWorldStateCollector(self.world, self.ego).collect()
        categories = {
            item["source_object_id"]: item["category"] for item in state["objects"]
        }
        self.assertEqual(categories["7"], "traffic_cone")
        self.assertEqual(categories["8"], "road_barrier")

    def test_classifies_adjacent_lane(self):
        left = FakeWaypoint(1, -2)
        ego = FakeWaypoint(1, -1, left=left)
        self.assertEqual(classify_lane_relation(ego, left), "left_adjacent_lane")

    def test_classifies_approaching_pedestrian_as_crossing_path(self):
        state = CarlaWorldStateCollector(self.world, self.ego).collect()
        walker = next(item for item in state["objects"] if item["source_object_id"] == "6")
        self.assertEqual(walker["lane_relation"], "crossing_ego_path")


if __name__ == "__main__":
    unittest.main()
