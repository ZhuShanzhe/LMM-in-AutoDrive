import carla
import random



class ObstacleSpawner:


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
    # 创建静态障碍物
    #
    # 例如:
    # - 路障
    # - 箱子
    # - 锥桶
    # - 坠落物
    #
    # 不负责:
    # - 出现时间
    # - 移动逻辑
    # =========================

    def spawn_obstacle(
        self,
        transform=None,
        obstacle_type=None
    ):



        if obstacle_type is None:


            obstacles = (
                self.blueprint_library
                .filter(
                    "static.prop.*"
                )
            )


            obstacle_bp = random.choice(
                obstacles
            )


        else:


            obstacle_bp = (
                self.blueprint_library
                .find(
                    obstacle_type
                )
            )



        if transform is None:


            transform = (
                self.get_random_spawn_point()
            )



        obstacle = (
            self.world.spawn_actor(
                obstacle_bp,
                transform
            )
        )


        return obstacle



    # =========================
    # 创建指定障碍物
    #
    # 更适合测试场景
    #
    # 例如:
    # traffic.cone01
    # static.prop.box01
    # =========================

    def spawn_specific_obstacle(
        self,
        obstacle_type,
        transform
    ):


        obstacle_bp = (
            self.blueprint_library
            .find(
                obstacle_type
            )
        )


        obstacle = (
            self.world.spawn_actor(
                obstacle_bp,
                transform
            )
        )


        return obstacle