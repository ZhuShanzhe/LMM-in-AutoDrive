import carla
import time

from perception.camera import CameraSensor
from perception.world_state import WorldState



def main():


    # ======================
    # 连接 CARLA
    # ======================

    client = carla.Client(
        "localhost",
        2000
    )


    client.set_timeout(
        10.0
    )


    world = client.get_world()



    vehicle = None

    camera = None



    try:


        # ======================
        # 创建 Ego Vehicle
        # ======================

        blueprint_library = (
            world
            .get_blueprint_library()
        )


        vehicle_bp = blueprint_library.find(
            "vehicle.tesla.model3"
        )


        spawn_points = (
            world
            .get_map()
            .get_spawn_points()
        )


        vehicle = world.spawn_actor(
            vehicle_bp,
            spawn_points[0]
        )


        print(
            "Vehicle:",
            vehicle.type_id
        )



        # ======================
        # Camera
        # ======================

        camera = CameraSensor(
            world,
            vehicle
        )


        camera.setup()


        print(
            "Camera setup finished"
        )



        # ======================
        # World State
        # ======================

        world_state = WorldState(
            world,
            vehicle
        )


        print(
            "WorldState setup finished"
        )



        # 等待camera第一帧

        time.sleep(2)



        # ======================
        # Observation
        # ======================

        for i in range(50):


            observation = {


                "image":
                camera.get_image(),


                "state":
                world_state.get_state()

            }



            print(
                "\n========== Observation =========="
            )


            if observation["image"] is not None:


                print(
                    "Image:",
                    observation["image"].shape
                )


            else:


                print(
                    "Image: None"
                )



            print(
                "State:"
            )


            print(
                observation["state"]
            )


            time.sleep(0.5)



    finally:


        print(
            "Cleaning..."
        )


        if camera:

            camera.destroy()


        if vehicle:

            vehicle.destroy()



        print(
            "Finished"
        )




if __name__ == "__main__":

    main()