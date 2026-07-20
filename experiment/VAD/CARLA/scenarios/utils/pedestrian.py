import carla
import random



class PedestrianSpawner:


    def __init__(self, world):

        self.world = world

        self.blueprint_library = (
            world.get_blueprint_library()
        )



    # ==================================
    # 创建行人
    # ==================================

    def spawn_pedestrian(
        self,
        transform=None,
        pedestrian_type=None
    ):



        if pedestrian_type:


            walker_bp = (
                self.blueprint_library
                .find(
                    pedestrian_type
                )
            )


        else:


            walkers = (
                self.blueprint_library
                .filter(
                    "walker.pedestrian.*"
                )
            )


            walker_bp = random.choice(
                walkers
            )



        # ------------------------------
        # 如果没有指定位置
        # 使用导航点
        # ------------------------------

        if transform is None:


            for _ in range(50):


                location = (
                    self.world
                    .get_random_location_from_navigation()
                )


                if location is None:

                    continue



                transform = carla.Transform(
                    location
                )


                actor = (
                    self.world
                    .try_spawn_actor(
                        walker_bp,
                        transform
                    )
                )


                if actor:

                    return actor



            raise RuntimeError(
                "Cannot spawn pedestrian"
            )



        # ------------------------------
        # 指定位置
        # 尝试多个高度
        # ------------------------------


        heights = [

            0.2,

            0.5,

            1.0

        ]



        for h in heights:


            new_transform = carla.Transform(

                carla.Location(

                    x=transform.location.x,

                    y=transform.location.y,

                    z=transform.location.z+h

                )

            )



            actor = (
                self.world
                .try_spawn_actor(
                    walker_bp,
                    new_transform
                )
            )


            if actor:

                return actor



        raise RuntimeError(
            "Pedestrian spawn failed"
        )



    # ==================================
    # Controller
    # ==================================

    def spawn_controller(
        self,
        pedestrian
    ):


        bp = (
            self.blueprint_library
            .find(
                "controller.ai.walker"
            )
        )


        controller = (
            self.world
            .spawn_actor(
                bp,
                carla.Transform(),
                attach_to=pedestrian
            )
        )


        return controller