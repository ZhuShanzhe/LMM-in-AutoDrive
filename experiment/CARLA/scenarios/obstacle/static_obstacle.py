import carla

from scenarios.base import BaseScenario
from scenarios.utils.vehicle import VehicleSpawner
from scenarios.utils.road import RoadFinder


class StaticObstacleScenario(BaseScenario):

    def __init__(
        self,
        world,
        external_control=True
    ):

        super().__init__(
            world,
            external_control
        )

        self.vehicle_spawner = VehicleSpawner(world)
        self.road_finder = RoadFinder(world)

        self.ego_vehicle = None
        self.obstacle = None

        self.collision_sensor = None
        self.collision_actor = None

        self.timeout = 30.0
        self.frame = 0

        self.obstacle_distance = 30

        self.goal_location = None
        self.route = []

        self.success_condition = {
            "type":
            "ego_reach_goal_without_collision"
        }

        self.failure_conditions = [
            "collision",
            "timeout"
        ]


    # =============================
    # collision
    # =============================

    def collision_callback(
        self,
        event
    ):

        other = event.other_actor

        self.collision_actor = other

        self.metrics["collision_count"] += 1

        print(
            "[StaticObstacle] Collision:",
            other.type_id
        )

        self.failure(
            "collision_with_" + other.type_id
        )



    # =============================
    # setup
    # =============================

    def setup(self):


        self.scenario_id = (
            "static_obstacle"
        )

        self.scenario_name = (
            "静态障碍物避让"
        )

        self.status = "RUNNING"



        self.trigger = {

            "type":
            "distance",

            "value":
            self.obstacle_distance,

            "description":
            "static obstacle ahead"

        }



        # =========================
        # 找道路
        # =========================

        road = (
            self.road_finder
            .find_straight_road(
                min_length=150
            )
        )


        road_wp = road["waypoint"]



        # =========================
        # ego
        # =========================

        ego_wp = road_wp.previous(40)[0]


        ego_tf = ego_wp.transform

        ego_tf.location.z += 0.1



        self.ego_vehicle = (
            self.vehicle_spawner
            .spawn_ego_vehicle(
                ego_tf
            )
        )


        if self.ego_vehicle is None:

            raise RuntimeError(
                "Ego spawn failed"
            )


        self.add_actor(
            self.ego_vehicle,
            "ego"
        )


        print(
            "[StaticObstacle] Ego:",
            self.ego_vehicle.id
        )



        # =========================
        # static obstacle
        # =========================

        obstacle_wp = (
            ego_wp.next(
                self.obstacle_distance
            )[0]
        )


        obstacle_tf = (
            obstacle_wp.transform
        )


        obstacle_tf.location.z += 0.05



        obstacle_bp = (
            self.world
            .get_blueprint_library()
            .find(
                "static.prop.trafficcone01"
            )
        )



        self.obstacle = (
            self.world.spawn_actor(
                obstacle_bp,
                obstacle_tf
            )
        )


        if self.obstacle is None:

            raise RuntimeError(
                "Obstacle spawn failed"
            )


        self.add_actor(
            self.obstacle,
            "obstacle"
        )


        print(
            "[StaticObstacle] Obstacle:",
            self.obstacle.id
        )



        # =========================
        # collision sensor
        # =========================

        collision_bp = (
            self.world
            .get_blueprint_library()
            .find(
                "sensor.other.collision"
            )
        )


        self.collision_sensor = (
            self.world.spawn_actor(
                collision_bp,
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



        # =========================
        # route
        # =========================

        self.build_route(
            ego_wp
        )


        print(
            "[StaticObstacle] Ready"
        )



    # =============================
    # route
    # =============================

    def build_route(
        self,
        ego_wp
    ):


        self.route = []

        current = ego_wp

        distance = 0


        while distance < 100:


            loc = (
                current.transform.location
            )


            self.route.append({

                "x":loc.x,
                "y":loc.y,
                "z":loc.z

            })


            nxt = current.next(5)


            if not nxt:
                break


            current = nxt[0]

            distance += 5



        if self.route:


            last = self.route[-1]

            self.goal_location = carla.Location(

                x=last["x"],
                y=last["y"],
                z=last["z"]

            )



    # =============================
    # tick
    # =============================

    def tick(self):


        if self._finished:

            return



        self.frame += 1


        self.metrics["simulation_time"] = (
            self.frame *
            self.fixed_delta_s
        )



        # timeout

        if (
            self.metrics["simulation_time"]
            >
            self.timeout
        ):

            self.failure(
                "timeout"
            )

            return



        ego_loc = (
            self.ego_vehicle
            .get_location()
        )


        obstacle_loc = (
            self.obstacle
            .get_location()
        )


        distance = (
            ego_loc.distance(
                obstacle_loc
            )
        )


        self.metrics["obstacle_distance_m"] = round(
            distance,
            3
        )



        # goal

        if self.goal_location:


            goal_distance = (
                ego_loc.distance(
                    self.goal_location
                )
            )


            self.metrics["goal_distance_m"] = round(
                goal_distance,
                3
            )


            if goal_distance < 3:


                self.success(
                    "ego_reached_goal"
                )



    # =============================
    # status
    # =============================

    def get_status(self):


        return {

            "scenario_id":
            self.scenario_id,

            "scenario_name":
            self.scenario_name,

            "status":
            self.status,

            "reason":
            self.reason,

            "actors":
            self.get_actor_ids(),

            "collision_actor":
            (
                None
                if self.collision_actor is None
                else self.collision_actor.id
            ),

            "metrics":
            self.metrics

        }



    def finished(self):

        return self._finished



    def get_ego_vehicle(self):

        return self.ego_vehicle