import carla

from scenarios.base import BaseScenario
from scenarios.utils.vehicle import VehicleSpawner
from scenarios.utils.road import RoadFinder


class CutInVehicleScenario(BaseScenario):

    def __init__(self, world, external_control=True):

        super().__init__(world, external_control)

        self.vehicle_spawner = VehicleSpawner(world)
        self.road_finder = RoadFinder(world)

        self.ego_vehicle = None
        self.cut_vehicle = None
        self.collision_sensor = None
        self.collision_actor = None

        self.frame = 0
        self.timeout = 30.0
        self.trigger_frame = 80

        self.cut_triggered = False
        self.cut_finished = False

        self.cut_target = None

        self.status = "RUNNING"
        self.reason = ""

        self.success_condition = {
            "type": "cut_in_completed_without_collision"
        }

        self.failure_conditions = [
            "collision",
            "timeout"
        ]


    def collision_callback(self, event):

        actor = event.other_actor
        self.collision_actor = actor

        self.metrics["collision_count"] += 1

        print(
            "[CutIn] Collision:",
            actor.type_id
        )

        self.failure(
            "collision_with_" + actor.type_id
        )


    def find_adjacent_lane(self, wp):

        for lane in [
            wp.get_left_lane(),
            wp.get_right_lane()
        ]:

            if lane and lane.lane_type == carla.LaneType.Driving:
                return lane

        return None



    def setup(self):

        self.scenario_id = "cut_in_vehicle"
        self.scenario_name = "车辆切入"

        road = self.road_finder.find_straight_road(
            min_length=150
        )

        road_wp = road["waypoint"]


        ego_wp_list = road_wp.previous(40)

        if not ego_wp_list:
            raise RuntimeError(
                "No ego waypoint"
            )

        ego_wp = ego_wp_list[0]


        ego_tf = ego_wp.transform
        ego_tf.location.z += 0.3


        self.ego_vehicle = (
            self.vehicle_spawner
            .spawn_ego_vehicle(
                ego_tf
            )
        )


        if self.ego_vehicle is None:
            raise RuntimeError(
                "ego spawn failed"
            )


        self.add_actor(
            self.ego_vehicle,
            "ego"
        )


        lane = self.find_adjacent_lane(
            ego_wp
        )


        if lane is None:
            raise RuntimeError(
                "No adjacent lane"
            )


        cut_points = lane.previous(15)

        if not cut_points:
            cut_points = lane.next(15)


        if not cut_points:
            raise RuntimeError(
                "No cut waypoint"
            )


        cut_tf = cut_points[0].transform
        cut_tf.location.z += 0.3


        bp = (
            self.world
            .get_blueprint_library()
            .find(
                "vehicle.tesla.model3"
            )
        )

        bp.set_attribute(
            "role_name",
            "cut_vehicle"
        )


        self.cut_vehicle = (
            self.world.try_spawn_actor(
                bp,
                cut_tf
            )
        )


        if self.cut_vehicle is None:
            raise RuntimeError(
                "cut vehicle spawn failed"
            )


        self.add_actor(
            self.cut_vehicle,
            "cut_vehicle"
        )


        self.cut_vehicle.set_autopilot(
            False
        )


        sensor_bp = (
            self.world
            .get_blueprint_library()
            .find(
                "sensor.other.collision"
            )
        )


        self.collision_sensor = (
            self.world.spawn_actor(
                sensor_bp,
                carla.Transform(),
                attach_to=self.ego_vehicle
            )
        )


        self.collision_sensor.listen(
            self.collision_callback
        )


        self.add_actor(
            self.collision_sensor,
            "collision_sensor"
        )


        print(
            "[CutIn] setup done"
        )



    def tick(self):

        if self._finished:
            return


        self.frame += 1

        self.metrics["simulation_time"] = (
            self.frame *
            self.fixed_delta_s
        )


        if self.metrics["simulation_time"] > self.timeout:

            self.failure(
                "timeout"
            )

            return


        if self.frame >= self.trigger_frame:

            self.cut_triggered = True


        if self.cut_triggered:

            self.execute_cut()



    def execute_cut(self):

        if self.cut_finished:
            return


        ego_tf = (
            self.ego_vehicle
            .get_transform()
        )

        cut_tf = (
            self.cut_vehicle
            .get_transform()
        )


        forward = ego_tf.get_forward_vector()

        right = ego_tf.get_right_vector()


        target = (
            ego_tf.location
            +
            forward * 20
        )


        self.cut_target = target


        diff = (
            target -
            cut_tf.location
        )

        diff.z = 0


        distance = diff.length()


        self.metrics["cut_distance_m"] = round(
            distance,
            3
        )


        if distance < 5:

            self.cut_finished = True

            self.success(
                "cut_in_completed"
            )

            return


        # 转换到车辆坐标系

        local_x = (
            diff.x * forward.x +
            diff.y * forward.y
        )

        local_y = (
            diff.x * right.x +
            diff.y * right.y
        )


        steer = 0

        if local_x > 0:

            steer = local_y / max(
                local_x,
                1
            )


        steer = max(
            -0.5,
            min(
                0.5,
                steer
            )
        )


        self.cut_vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.45,
                steer=steer
            )
        )


    def get_status(self):

        return {

            "status": self.status,

            "reason": self.reason,

            "actors": self.get_actor_ids(),

            "collision_actor":
            (
                None
                if self.collision_actor is None
                else self.collision_actor.id
            ),

            "cut_triggered":
            self.cut_triggered,

            "cut_finished":
            self.cut_finished,

            "metrics":
            self.metrics
        }



    def finished(self):

        return self._finished



    def get_ego_vehicle(self):

        return self.ego_vehicle