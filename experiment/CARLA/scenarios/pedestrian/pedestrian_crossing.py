import carla


from scenarios.base import BaseScenario

from scenarios.utils.vehicle import VehicleSpawner

from scenarios.utils.pedestrian import PedestrianSpawner





class PedestrianCrossingScenario(BaseScenario):


    def __init__(
        self,
        world,
        external_control=True
    ):


        super().__init__(
            world,
            external_control
        )



        self.vehicle_spawner = VehicleSpawner(
            world
        )


        self.pedestrian_spawner = PedestrianSpawner(
            world
        )



        self.ego_vehicle = None


        self.walker = None



        self.collision_sensor = None


        self.collision_actor = None



        self.cross_direction = None



        self.walker_goal = None



        self.frame = 0



        self.timeout = 30.0






    # ==================================
    # collision
    # ==================================

    def collision_callback(
        self,
        event
    ):


        other_actor = event.other_actor


        self.collision_actor = other_actor


        self.metrics["collision_count"] += 1



        print(
            "[Pedestrian] Collision:",
            other_actor.type_id
        )



        self.failure(

            "collision_with_"
            +
            other_actor.type_id

        )







    # ==================================
    # setup
    # ==================================

    def setup(self):


        self.scenario_id = (
            "pedestrian_crossing"
        )


        self.scenario_name = (
            "行人横穿道路"
        )


        self.status = "RUNNING"





        self.trigger = {


            "type":

            "walker_crossing",


            "description":

            "walker crosses ego lane"

        }



        self.failure_conditions = [

            "collision_with_walker",

            "timeout"

        ]







        spawn_points = (

            self.world
            .get_map()
            .get_spawn_points()

        )



        if len(spawn_points)==0:


            raise RuntimeError(

                "No spawn points"

            )



        ego_transform = spawn_points[0]






        # ==========================
        # ego
        # ==========================


        self.ego_vehicle = (

            self.vehicle_spawner
            .spawn_ego_vehicle(

                ego_transform

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

            "[Pedestrian] Ego:",

            self.ego_vehicle.id

        )
        # ==========================
        # 横穿方向
        # ==========================


        ego_right = (

            ego_transform
            .get_right_vector()

        )



        self.cross_direction = carla.Vector3D(

            -ego_right.x,

            -ego_right.y,

            0

        )






        # ==========================
        # ego前方25m
        # 右侧5m
        # ==========================


        ego_forward = (

            ego_transform
            .get_forward_vector()

        )



        front_distance = 25


        side_distance = 5





        walker_start = carla.Location(

            x =
            ego_transform.location.x
            +
            ego_forward.x * front_distance
            +
            ego_right.x * side_distance,


            y =
            ego_transform.location.y
            +
            ego_forward.y * front_distance
            +
            ego_right.y * side_distance,


            z =
            ego_transform.location.z + 0.5

        )






        # ==========================
        # walker终点
        # ==========================


        walker_end = carla.Location(

            x =
            walker_start.x
            +
            self.cross_direction.x * 10,


            y =
            walker_start.y
            +
            self.cross_direction.y * 10,


            z =
            walker_start.z

        )


        self.walker_goal = walker_end







        walker_transform = carla.Transform(


            walker_start,


            carla.Rotation(

                yaw =
                ego_transform.rotation.yaw - 90

            )

        )






        # ==========================
        # 创建walker
        # ==========================


        self.walker = (

            self.pedestrian_spawner
            .spawn_pedestrian(

                walker_transform

            )

        )



        if self.walker is None:


            raise RuntimeError(

                "Walker spawn failed"

            )



        self.add_actor(

            self.walker,

            "walker"

        )








        # ==========================
        # collision sensor
        # ==========================


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








        # ==========================
        # route
        # ==========================


        self.route = []


        current_wp = (

            self.world
            .get_map()
            .get_waypoint(

                ego_transform.location,

                project_to_road=True

            )

        )



        distance = 0



        while distance < 100:


            loc = (

                current_wp
                .transform
                .location

            )



            self.route.append(

                {

                    "x":loc.x,

                    "y":loc.y,

                    "z":loc.z

                }

            )



            next_wp = current_wp.next(5)



            if len(next_wp)==0:

                break



            current_wp = next_wp[0]


            distance += 5





        if len(self.route)>0:


            last = self.route[-1]


            self.goal_location = carla.Location(

                x=last["x"],

                y=last["y"],

                z=last["z"]

            )






        print(

            "[Pedestrian] Walker:",

            self.walker.id

        )


        print(

            "[Pedestrian] Goal:",

            self.goal_location

        )


        print(

            "[Pedestrian] Ready"

        )







    # ==================================
    # tick
    # ==================================

    def tick(self):


        if self._finished:

            return



        self.frame += 1



        self.metrics["simulation_time"] = (

            self.frame * 0.05

        )





        # ==========================
        # timeout
        # ==========================


        if (

            self.metrics["simulation_time"]

            >

            self.timeout

        ):

            self.failure(

                "timeout"

            )

            return







        # ==========================
        # walker移动
        # ==========================


        if self.walker is not None:



            control = carla.WalkerControl()



            control.direction = (

                self.cross_direction

            )


            control.speed = 1.2



            self.walker.apply_control(

                control

            )







        # ==========================
        # ego到达终点
        # ==========================


        if self.goal_location is not None:



            ego_location = (

                self.ego_vehicle
                .get_location()

            )



            distance = ego_location.distance(

                self.goal_location

            )



            if distance < 3.0:



                self.success(

                    "ego_reached_goal"

                )









    # ==================================
    # status
    # ==================================

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

        }







    def finished(self):

        return self._finished







    def get_ego_vehicle(self):

        return self.ego_vehicle