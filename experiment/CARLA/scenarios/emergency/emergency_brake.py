import carla


from scenarios.base import BaseScenario

from scenarios.utils.vehicle import VehicleSpawner

from scenarios.utils.road import RoadFinder





class EmergencyBrakeScenario(BaseScenario):


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


        self.road_finder = RoadFinder(
            world
        )



        self.ego_vehicle = None

        self.front_vehicle = None


        self.collision_sensor = None


        self.collision_actor = None




        # =====================
        # 场景参数
        # =====================

        # 初始车距

        self.initial_distance = 15



        # 前车速度

        self.front_speed = 30



        # 前车巡航油门

        self.front_throttle = 0.35




        # =====================
        # 急刹参数
        # =====================


        self.frame = 0


        self.brake_trigger_frame = 100



        self.brake_triggered = False






        # =====================
        # 状态
        # =====================


        self.status = "RUNNING"


        self.reason = ""



        self.timeout = 30.0



        # 成功判断

        self.goal_location = None

        self.safe_stop_speed_kmh = 0.5
        self.safe_stop_hold_s = 1.0
        self.min_safe_gap_m = 5.0
        self.min_trigger_speed_kmh = 3.0
        self.safe_stop_elapsed_s = 0.0
        self.brake_trigger_time_s = None
        self.success_condition = {
            "type": "safe_stop_after_front_brake",
            "max_ego_speed_kmh": self.safe_stop_speed_kmh,
            "hold_s": self.safe_stop_hold_s,
            "min_front_gap_m": self.min_safe_gap_m,
        }
        self.failure_conditions = [
            "collision",
            "timeout",
            "unsafe_stop_gap",
            "ego_not_moving_at_brake_trigger",
            "goal_reached_without_safe_stop",
        ]






    # =================================
    # collision
    # =================================

    def collision_callback(
        self,
        event
    ):


        other_actor = event.other_actor


        self.collision_actor = other_actor


        self.metrics["collision_count"] += 1



        print(
            "[EmergencyBrake] collision:",
            other_actor.type_id
        )



        self.failure(

            "collision_with_"
            +
            other_actor.type_id

        )







    # =================================
    # setup
    # =================================

    def setup(self):


        self.scenario_id = (
            "emergency_brake"
        )


        self.scenario_name = (
            "前车紧急制动"
        )


        self.status = "RUNNING"




        self.trigger = {


            "type":
            "time",


            "value":
            5,


            "description":
            "front vehicle brake after 5 seconds"

        }






        # =====================
        # 找直道路
        # =====================


        road = (

            self.road_finder
            .find_straight_road(

                min_length=120

            )

        )


        road_wp = road["waypoint"]







        # =====================
        # 从已验证的直路起点出发
        # =====================


        ego_wp = road_wp



        ego_transform = ego_wp.transform


        ego_transform.location.z +=0.3








        # =====================
        # ego vehicle
        # =====================


        ego_bp = (

            self.world
            .get_blueprint_library()
            .find(

                "vehicle.tesla.model3"

            )

        )



        ego_bp.set_attribute(

            "role_name",

            "ego"

        )



        self.ego_vehicle = (

            self.world
            .try_spawn_actor(

                ego_bp,

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






        # =====================
        # front vehicle
        # =====================


        front_candidates = (

            ego_wp.next(

                self.initial_distance

            )

        )



        if len(front_candidates)==0:


            raise RuntimeError(

                "Cannot find front waypoint"

            )



        front_transform = (

            front_candidates[0]
            .transform

        )



        front_transform.location.z +=0.3







        front_bp = (

            self.world
            .get_blueprint_library()
            .find(

                "vehicle.tesla.model3"

            )

        )



        front_bp.set_attribute(

            "role_name",

            "front_vehicle"

        )



        self.front_vehicle = (

            self.world
            .try_spawn_actor(

                front_bp,

                front_transform

            )

        )



        if self.front_vehicle is None:


            raise RuntimeError(

                "Front vehicle spawn failed"

            )



        self.add_actor(

            self.front_vehicle,

            "front_vehicle"

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
        # 生成ego目标终点
        # =====================

        self.route = []


        current_wp = ego_wp


        distance = 0



        while distance < 100:


            loc = (

                current_wp
                .transform
                .location

            )


            self.route.append(

                {

                    "x": loc.x,

                    "y": loc.y,

                    "z": loc.z

                }

            )



            next_wps = current_wp.next(5)



            if len(next_wps) == 0:

                break



            current_wp = next_wps[0]


            distance += 5






        if len(self.route) > 0:


            last = self.route[-1]


            self.goal_location = carla.Location(

                x=last["x"],

                y=last["y"],

                z=last["z"]

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

                    x=-10,

                    z=6

                ),


                carla.Rotation(

                    pitch=-25,

                    yaw=ego_tf.rotation.yaw

                )

            )

        )





        print(
            "[EmergencyBrake] setup done"
        )


        print(
            "ego id:",
            self.ego_vehicle.id
        )


        print(
            "front id:",
            self.front_vehicle.id
        )


        print(
            "goal:",
            self.goal_location
        )








    # =================================
    # tick
    # =================================

    def tick(self):


        if self._finished:

            return



        if self.front_vehicle is None:

            return





        # =====================
        # 更新时间
        # =====================

        self.metrics["simulation_time"] += self.fixed_delta_s






        # =====================
        # timeout
        # =====================

        if (

            self.metrics["simulation_time"]

            >

            self.timeout

        ):


            self.failure(

                "timeout"

            )


            return




        # =====================
        # 前车巡航
        # =====================

        if not self.brake_triggered:


            self.front_vehicle.apply_control(

                carla.VehicleControl(

                    throttle=self.front_throttle,

                    brake=0.0

                )

            )







        # =====================
        # 急刹触发
        # =====================

        self.frame += 1



        if (

            self.frame >= self.brake_trigger_frame

            and

            not self.brake_triggered

        ):


            print(

                "[EmergencyBrake] brake!"

            )



            self.front_vehicle.apply_control(

                carla.VehicleControl(

                    throttle=0.0,

                    brake=1.0

                )

            )



            self.brake_triggered = True
            self.brake_trigger_time_s = self.metrics["simulation_time"]
            self.metrics["brake_triggered"] = True
            ego_velocity = self.ego_vehicle.get_velocity()
            trigger_speed_kmh = 3.6 * (
                ego_velocity.x ** 2
                + ego_velocity.y ** 2
                + ego_velocity.z ** 2
            ) ** 0.5
            self.metrics["ego_speed_at_brake_trigger_kmh"] = round(
                trigger_speed_kmh, 3
            )
            if trigger_speed_kmh < self.min_trigger_speed_kmh:
                self.failure("ego_not_moving_at_brake_trigger")
                return

        if self.brake_triggered:
            ego_velocity = self.ego_vehicle.get_velocity()
            ego_speed_kmh = 3.6 * (
                ego_velocity.x ** 2 + ego_velocity.y ** 2 + ego_velocity.z ** 2
            ) ** 0.5
            front_distance = self.ego_vehicle.get_location().distance(
                self.front_vehicle.get_location()
            )
            previous_min = self.metrics.get("min_distance")
            self.metrics["min_distance"] = (
                front_distance if previous_min is None else min(previous_min, front_distance)
            )
            self.metrics["front_distance_m"] = round(front_distance, 3)
            self.metrics["ego_speed_kmh"] = round(ego_speed_kmh, 3)
            if self.metrics.get("reaction_time") is None:
                applied_control = self.ego_vehicle.get_control()
                if applied_control.brake >= 0.2:
                    self.metrics["reaction_time"] = round(
                        self.metrics["simulation_time"]
                        - self.brake_trigger_time_s,
                        3,
                    )

            if ego_speed_kmh <= self.safe_stop_speed_kmh:
                self.safe_stop_elapsed_s += self.fixed_delta_s
                if self.safe_stop_elapsed_s >= self.safe_stop_hold_s:
                    if front_distance >= self.min_safe_gap_m:
                        self.success("safe_stop_after_front_brake")
                    else:
                        self.failure("unsafe_stop_gap")
                    return
            else:
                self.safe_stop_elapsed_s = 0.0






        # =====================
        # ego到达终点
        # =====================


        if self.goal_location is not None:



            ego_location = (

                self.ego_vehicle
                .get_location()

            )


            distance = ego_location.distance(

                self.goal_location

            )



            if distance < 3.0:
                self.failure(
                    "goal_reached_without_safe_stop"
                )









    # =================================
    # status
    # =================================

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

            "brake_triggered": self.brake_triggered,

            "metrics": self.metrics

        }







    # =================================
    # finished
    # =================================

    def finished(self):

        return self._finished






    # =================================
    # ego接口
    # =================================

    def get_ego_vehicle(self):

        return self.ego_vehicle
