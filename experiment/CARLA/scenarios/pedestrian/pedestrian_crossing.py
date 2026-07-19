import carla

from scenarios.base import BaseScenario

from scenarios.utils.vehicle import VehicleSpawner
from scenarios.utils.pedestrian import PedestrianSpawner



class PedestrianCrossingScenario(BaseScenario):


    def __init__(self, world):

        super().__init__(world)


        self.vehicle_spawner = VehicleSpawner(
            world
        )


        self.pedestrian_spawner = PedestrianSpawner(
            world
        )


        self.ego_vehicle = None

        self.walker = None


        # 行人横穿方向

        self.cross_direction = None



    # ==================================
    # Ego保持前进
    # ==================================

    def keep_ego_move(self):


        if self.ego_vehicle is None:

            return



        transform = (
            self.ego_vehicle
            .get_transform()
        )


        forward = (
            transform
            .get_forward_vector()
        )



        speed = 5.0



        self.ego_vehicle.set_target_velocity(

            carla.Vector3D(

                forward.x * speed,

                forward.y * speed,

                0

            )

        )



    # ==================================
    # 初始化场景
    # ==================================

    def setup(self):


        spawn_points = (
            self.world
            .get_map()
            .get_spawn_points()
        )


        ego_transform = spawn_points[0]



        # ------------------------------
        # 创建 Ego
        # ------------------------------

        self.ego_vehicle = (

            self.vehicle_spawner
            .spawn_ego_vehicle(

                ego_transform

            )

        )


        self.add_actor(
            self.ego_vehicle
        )



        print(
            "[Pedestrian] Ego:",
            self.ego_vehicle.id
        )



        # ------------------------------
        # 计算横穿方向
        # ------------------------------

        ego_right = (

            ego_transform
            .get_right_vector()

        )



        # 注意：
        # 取反，避免行人反向移动

        self.cross_direction = carla.Vector3D(

            -ego_right.x,

            -ego_right.y,

            0

        )



        # ------------------------------
        # 行人生成位置
        #
        # Ego前方25m
        # 右侧5m
        #
        # ------------------------------

        ego_forward = (

            ego_transform
            .get_forward_vector()

        )



        front_distance = 25

        side_distance = 5



        walker_location = carla.Location(

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



        # ------------------------------
        # 行人朝向横穿方向
        # ------------------------------

        walker_yaw = (

            ego_transform.rotation.yaw
            -
            90

        )



        walker_transform = carla.Transform(

            walker_location,

            carla.Rotation(

                yaw=walker_yaw

            )

        )



        # ------------------------------
        # 创建行人
        # ------------------------------

        self.walker = (

            self.pedestrian_spawner
            .spawn_pedestrian(

                walker_transform

            )

        )


        self.add_actor(
            self.walker
        )



        print(

            "[Pedestrian] Walker:",

            self.walker.id

        )



        self.world.tick()



        print(
            "[Pedestrian] Ready"
        )



    # ==================================
    # 每帧更新
    # ==================================

    def tick(self):


        # Ego

        self.keep_ego_move()



        # ------------------------------
        # 行人横穿
        # ------------------------------

        if self.walker:


            control = carla.WalkerControl()



            control.direction = (

                self.cross_direction

            )



            control.speed = 1.2



            self.walker.apply_control(

                control

            )



    # ==================================
    # 场景结束
    # ==================================

    def finished(self):

        return False



    def get_ego_vehicle(self):

        return self.ego_vehicle