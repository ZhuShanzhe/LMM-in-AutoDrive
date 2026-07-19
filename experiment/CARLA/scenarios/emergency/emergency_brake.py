import carla

from scenarios.base import BaseScenario

from scenarios.utils.vehicle import VehicleSpawner



class EmergencyBrakeScenario(BaseScenario):


    def __init__(
        self,
        world
    ):

        super().__init__(
            world
        )


        self.vehicle_spawner = (
            VehicleSpawner(
                world
            )
        )


        self.ego_vehicle = None

        self.front_vehicle = None


        self.frame = 0


        # NPC急刹时间
        self.brake_trigger_frame = 200



        # 初始速度 km/h

        self.initial_speed = 30



    # =================================
    # 设置车辆速度
    #
    # 使用CARLA 0.9.16:
    # set_target_velocity
    # =================================

    def set_vehicle_speed(
        self,
        vehicle,
        speed_kmh
    ):


        speed_ms = (
            speed_kmh / 3.6
        )


        transform = (
            vehicle.get_transform()
        )


        forward = (
            transform
            .get_forward_vector()
        )


        velocity = carla.Vector3D(

            x=forward.x * speed_ms,

            y=forward.y * speed_ms,

            z=0

        )


        vehicle.set_target_velocity(
            velocity
        )



    # =================================
    # 根据Ego生成前车位置
    #
    # Ego前方distance米
    # =================================

    def get_front_spawn(
        self,
        ego_transform,
        distance=25
    ):


        forward = (
            ego_transform
            .get_forward_vector()
        )


        location = (
            ego_transform.location
            +
            carla.Location(

                x=forward.x * distance,

                y=forward.y * distance,

                z=0

            )
        )


        return carla.Transform(

            location,

            ego_transform.rotation

        )



    # =================================
    # 初始化场景
    # =================================

    def setup(self):


        spawn_points = (
            self.world
            .get_map()
            .get_spawn_points()
        )


        if len(spawn_points) == 0:

            raise RuntimeError(
                "No spawn points"
            )



        # =============================
        # Ego生成
        # =============================

        ego_spawn = (
            spawn_points[0]
        )


        self.ego_vehicle = (
            self.vehicle_spawner
            .spawn_ego_vehicle(
                ego_spawn
            )
        )


        self.add_actor(
            self.ego_vehicle
        )



        # Ego初始速度

        self.set_vehicle_speed(
            self.ego_vehicle,
            self.initial_speed
        )



        # =============================
        # 前车生成
        # =============================

        front_spawn = (
            self.get_front_spawn(
                ego_spawn,
                25
            )
        )



        self.front_vehicle = (
            self.vehicle_spawner
            .spawn_npc_vehicle(

                front_spawn,

                "vehicle.tesla.model3"

            )
        )


        self.add_actor(
            self.front_vehicle
        )



        # NPC初始速度

        self.set_vehicle_speed(
            self.front_vehicle,
            self.initial_speed
        )



        print(
            "[EmergencyBrake]"
            " Ego:",
            self.ego_vehicle.id
        )


        print(
            "[EmergencyBrake]"
            " Front:",
            self.front_vehicle.id
        )


        print(
            "[EmergencyBrake]"
            " Initial speed:",
            self.initial_speed,
            "km/h"
        )



    # =================================
    # 场景更新
    #
    # 这里只改变NPC
    #
    # 不控制Ego
    # =================================

    def tick(self):


        self.frame += 1



        if (
            self.frame ==
            self.brake_trigger_frame
        ):


            print(
                "[EmergencyBrake]"
                " NPC emergency brake!"
            )


            # 解除速度目标

            self.front_vehicle.set_target_velocity(

                carla.Vector3D(
                    0,
                    0,
                    0
                )

            )



            # NPC刹车

            self.front_vehicle.apply_control(

                carla.VehicleControl(

                    throttle=0.0,

                    brake=1.0

                )

            )



    # =================================
    # 当前测试不自动结束
    # =================================

    def finished(self):

        return False



    # =================================
    # Ego接口
    # 提供给感知/决策模块
    # =================================

    def get_ego_vehicle(self):

        return self.ego_vehicle