import carla


from scenarios.base import BaseScenario
from scenarios.utils.vehicle import VehicleSpawner





class StraightDrivingScenario(BaseScenario):


    def __init__(
        self,
        world,
        external_control=True
    ):


        super().__init__(
            world,
            external_control
        )


        self.scenario_id = (
            "straight_driving"
        )


        self.scenario_name = (
            "Straight Driving"
        )


        self.vehicle_spawner = VehicleSpawner(
            world
        )


        self.ego_vehicle = None

        self.camera = None


        self.collision_sensor = None

        self.collision_actor = None



        self.route = []

        self.goal_location = None



        self.timeout = 30.0

        self.goal_tolerance_m = 3.0
        self.max_lateral_offset_m = 1.0
        self.route_forward = None
        self.success_condition = {
            "type": "goal_with_lane_centering",
            "goal_tolerance_m": self.goal_tolerance_m,
            "max_lateral_offset_m": self.max_lateral_offset_m,
        }
        self.failure_conditions = ["collision", "timeout"]







    # =========================
    # collision callback
    # =========================

    def collision_callback(
        self,
        event
    ):


        other_actor = event.other_actor


        self.collision_actor = other_actor


        self.metrics["collision_count"] += 1



        print(
            "[StraightDriving] Collision:",
            other_actor.type_id
        )



        self.failure(

            "collision_with_"
            +
            other_actor.type_id

        )








    # =========================
    # setup
    # =========================

    def setup(self):


        # =====================
        # 原始车辆位置
        # =====================


        base_transform = carla.Transform(

            carla.Location(

                x=-52.073921,

                y=100.189049,

                z=0.6

            ),


            carla.Rotation(

                yaw=89.83876

            )

        )





        # =====================
        # 右侧车道偏移
        # =====================


        right_vector = (

            base_transform
            .get_right_vector()

        )



        lane_offset = -7



        ego_location = (

            base_transform.location

            +

            carla.Location(

                x=right_vector.x * lane_offset,

                y=right_vector.y * lane_offset,

                z=0

            )

        )





        # =====================
        # 车头旋转180°
        # =====================


        ego_rotation = carla.Rotation(

            pitch=0,

            roll=0,

            yaw=

            base_transform.rotation.yaw

            +

            180

        )



        ego_spawn = carla.Transform(

            ego_location,

            ego_rotation

        )


        ego_spawn.location.z += 0.3







        # =====================
        # ego
        # =====================


        self.ego_vehicle = (

            self.vehicle_spawner
            .spawn_ego_vehicle(

                ego_spawn

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

            "[StraightDriving] Ego:",

            self.ego_vehicle.id

        )








        # =====================
        # collision sensor
        # =====================


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









        # =====================
        # RGB Camera
        # =====================


        camera_bp = (

            self.world
            .get_blueprint_library()
            .find(

                "sensor.camera.rgb"

            )

        )



        camera_bp.set_attribute(

            "image_size_x",

            "1280"

        )


        camera_bp.set_attribute(

            "image_size_y",

            "720"

        )


        camera_bp.set_attribute(

            "fov",

            "90"

        )





        camera_transform = carla.Transform(

            carla.Location(

                x=1.5,

                y=0,

                z=2.2

            ),


            carla.Rotation(

                pitch=-10

            )

        )





        self.camera = (

            self.world.spawn_actor(

                camera_bp,

                camera_transform,

                attach_to=self.ego_vehicle

            )

        )



        self.add_actor(

            self.camera,

            "camera"

        )



        print(

            "[StraightDriving] Camera:",

            self.camera.id

        )









        # =====================
        # spectator
        # =====================


        spectator = (

            self.world
            .get_spectator()

        )



        ego_tf = (

            self.ego_vehicle
            .get_transform()

        )



        spectator.set_transform(


            carla.Transform(


                ego_tf.location

                +

                carla.Location(

                    x=-8,

                    z=5

                ),



                carla.Rotation(

                    pitch=-20,

                    yaw=ego_tf.rotation.yaw

                )

            )

        )







        # =====================
        # route
        # =====================


        self.build_route(

            ego_spawn

        )



        self.status = "RUNNING"



        print(

            "[StraightDriving] Ready"

        )







    # =========================
    # route
    # =========================

    def build_route(

        self,

        transform

    ):


        location = transform.location


        forward = (

            transform
            .get_forward_vector()

        )

        self.route_forward = forward



        distance = 0



        while distance < 100:



            self.route.append(


                {

                    "x":

                    location.x

                    +

                    forward.x * distance,


                    "y":

                    location.y

                    +

                    forward.y * distance,


                    "z":

                    location.z


                }

            )



            distance += 5






        if len(self.route)>0:


            last = self.route[-1]



            self.goal_location = carla.Location(

                x=last["x"],

                y=last["y"],

                z=last["z"]

            )









    # =========================
    # tick
    # =========================

    def tick(self):


        if self._finished:

            return




        self.metrics["simulation_time"] += self.fixed_delta_s





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





        # ego到终点


        if self.goal_location is not None:


            ego_location = (

                self.ego_vehicle
                .get_location()

            )



            distance = ego_location.distance(

                self.goal_location

            )



            delta_x = ego_location.x - self.goal_location.x
            delta_y = ego_location.y - self.goal_location.y
            if self.route_forward is None:
                lateral_offset = 0.0
            else:
                lateral_offset = abs(
                    delta_x * -self.route_forward.y + delta_y * self.route_forward.x
                )
            self.metrics["goal_distance_m"] = round(distance, 3)
            self.metrics["lateral_offset_m"] = round(lateral_offset, 3)

            if distance < self.goal_tolerance_m and lateral_offset <= self.max_lateral_offset_m:


                self.success(

                    "ego_reached_goal_centered"

                )









    # =========================
    # status
    # =========================

    def get_status(self):


        return {


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

                else

                self.collision_actor.id

            )

            ,

            "metrics": self.metrics

        }







    def finished(self):


        return self._finished






    def get_ego_vehicle(self):


        return self.ego_vehicle
