import carla
import time
import sys


# =========================
# 导入场景
# =========================

from scenarios.basic.straight_driving import (
    StraightDrivingScenario
)


from scenarios.emergency.emergency_brake import (
    EmergencyBrakeScenario
)

from scenarios.pedestrian.pedestrian_crossing import (
    PedestrianCrossingScenario
)

# =========================
# 场景注册表
# =========================

SCENARIOS = {


    "straight_driving":
        StraightDrivingScenario,


    "emergency_brake":
        EmergencyBrakeScenario,

    "pedestrian_crossing":
        PedestrianCrossingScenario,

}



def print_available_scenarios():


    print("\nAvailable scenarios:")


    for name in SCENARIOS.keys():

        print(
            " -",
            name
        )



def main():


    # =====================
    # 检查参数
    # =====================

    if len(sys.argv) < 2:


        print(
            "Please specify scenario name."
        )


        print_available_scenarios()

        return



    scenario_name = sys.argv[1]



    if scenario_name not in SCENARIOS:


        print(
            "Unknown scenario:",
            scenario_name
        )


        print_available_scenarios()

        return



    # =====================
    # 连接CARLA
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

    scenario_class = (
        SCENARIOS[scenario_name]
    )


    scenario = scenario_class(
        world
    )



    try:


        print(
            "\n[Scenario]",
            scenario_name
        )



        # 初始化

        scenario.setup()

        print(scenario.get_scenario_info()) # 获取信息

        print(
            "[Scenario] Running..."
        )



        # =====================
        # 主循环
        # =====================

        while not scenario.finished():


            # 更新场景

            scenario.tick()



            # 推进CARLA

            world.tick()



            time.sleep(
                0.05
            )



    except KeyboardInterrupt:


        print(
            "\n[Scenario] Interrupted"
        )



    finally:

        print(scenario.get_scenario_info())

        print(
            "[Scenario] Cleaning..."
        )


        scenario.destroy()



        print(
            "[Scenario] Finished"
        )




if __name__ == "__main__":

    main()