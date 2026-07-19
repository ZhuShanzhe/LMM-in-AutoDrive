import carla
import random



class VehicleSpawner:


    def __init__(
        self,
        world
    ):

        self.world = world

        self.blueprint_library = (
            world.get_blueprint_library()
        )



    # =========================
    # 获取随机生成点
    # =========================

    def get_random_spawn_point(self):


        spawn_points = (
            self.world
            .get_map()
            .get_spawn_points()
        )


        if len(spawn_points) == 0:

            raise RuntimeError(
                "No spawn points available"
            )


        return random.choice(
            spawn_points
        )



    # =========================
    # 创建 Ego车辆
    #
    # 只负责:
    # - blueprint
    # - spawn
    #
    # 不负责:
    # - 速度
    # - 控制
    # =========================

    def spawn_ego_vehicle(
        self,
        spawn_point=None,
        vehicle_type="vehicle.tesla.model3"
    ):


        vehicle_bp = (
            self.blueprint_library
            .find(
                vehicle_type
            )
        )


        vehicle_bp.set_attribute(
            "role_name",
            "ego"
        )



        if spawn_point is None:


            spawn_point = (
                self.get_random_spawn_point()
            )



        vehicle = (
            self.world.spawn_actor(
                vehicle_bp,
                spawn_point
            )
        )


        return vehicle



    # =========================
    # 创建 NPC车辆
    #
    # 用于:
    # - 前车
    # - 后车
    # - 旁车
    #
    # 不负责:
    # - 速度
    # - 行为
    # =========================

    def spawn_npc_vehicle(
        self,
        spawn_point=None,
        vehicle_type=None
    ):



        if vehicle_type is None:


            vehicle_bps = (
                self.blueprint_library
                .filter(
                    "vehicle.*"
                )
            )


            vehicle_bp = random.choice(
                vehicle_bps
            )


        else:


            vehicle_bp = (
                self.blueprint_library
                .find(
                    vehicle_type
                )
            )



        if vehicle_bp.has_attribute(
            "role_name"
        ):

            vehicle_bp.set_attribute(
                "role_name",
                "npc"
            )



        if spawn_point is None:


            spawn_point = (
                self.get_random_spawn_point()
            )



        vehicle = (
            self.world.spawn_actor(
                vehicle_bp,
                spawn_point
            )
        )


        return vehicle