import carla
import time


from scenarios.emergency.emergency_brake import (
    EmergencyBrakeScenario
)


from perception.ground_truth import (
    GroundTruth
)



def main():


    # =====================
    # CARLA连接
    # =====================

    client = carla.Client(
        "localhost",
        2000
    )


    client.set_timeout(
        10.0
    )


    world = client.get_world()



    # =====================
    # 创建场景
    # =====================

    scenario = EmergencyBrakeScenario(
        world
    )


    try:


        scenario.setup()



        # =====================
        # 创建感知模块
        # Ego由scenario创建
        # =====================

        perception = GroundTruth(
            world,
            scenario.ego_vehicle
        )


        print(
            "\n[System] Start perception loop\n"
        )



        # =====================
        # 主循环
        # =====================

        while not scenario.finished():


            # 更新场景

            scenario.tick()



            # 推进仿真

            world.tick()



            # =====================
            # 获取真值
            # =====================

            ego_state = (
                perception.get_state()
            )


            nearby_vehicle = (
                perception
                .get_nearby_vehicles(
                    max_distance=100
                )
            )



            print(
                "\n=========="
            )


            print(
                "Ego state:"
            )


            print(
                ego_state
            )


            print(
                "Nearby vehicles:"
            )


            print(
                nearby_vehicle
            )



            time.sleep(
                0.1
            )



    except KeyboardInterrupt:


        print(
            "\n[System] Interrupted"
        )



    finally:


        print(
            "\n[System] Cleaning..."
        )


        scenario.destroy()



        print(
            "[System] Finished"
        )




if __name__ == "__main__":

    main()