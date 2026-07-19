from scenarios.base import BaseScenario

from scenarios.utils.vehicle import VehicleSpawner



class StraightDrivingScenario(BaseScenario):


    def __init__(
        self,
        world
    ):

        super().__init__(
            world
        )


        self.ego_vehicle = None


        self.vehicle_spawner = (
            VehicleSpawner(
                world
            )
        )



    # =========================
    # 初始化场景
    #
    # 负责:
    # - 创建Ego
    # - 设置初始环境
    #
    # 不负责:
    # - 控制车辆
    # =========================

    def setup(self):


        spawn_points = (
            self.world
            .get_map()
            .get_spawn_points()
        )


        if len(spawn_points) == 0:

            raise RuntimeError(
                "No spawn points available"
            )



        ego_spawn = (
            spawn_points[0]
        )



        # 创建Ego车辆

        self.ego_vehicle = (
            self.vehicle_spawner
            .spawn_ego_vehicle(
                ego_spawn
            )
        )



        self.add_actor(
            self.ego_vehicle
        )



        print(
            "[StraightDriving]"
            " Ego vehicle created:",
            self.ego_vehicle.id
        )


        print(
            "[StraightDriving]"
            " Waiting for vehicle controller"
        )



    # =========================
    # 每帧更新
    #
    # 当前:
    # 不控制Ego
    #
    # 后续:
    # 环境变化
    # NPC行为
    # =========================

    def tick(self):

        pass



    # =========================
    # 提供Ego接口
    #
    # 给:
    # - perception
    # - decision
    # - controller
    #
    # =========================

    def get_ego_vehicle(self):

        return self.ego_vehicle