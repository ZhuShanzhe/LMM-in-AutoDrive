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

from scenarios.emergency.cut_in_vehicle import (
    CutInVehicleScenario
)

from scenarios.pedestrian.pedestrian_crossing import (
    PedestrianCrossingScenario
)

from scenarios.obstacle.static_obstacle import (
    StaticObstacleScenario
)


# =========================
# 场景注册表
# =========================

SCENARIOS = {

    "straight_driving":
        StraightDrivingScenario,


    "emergency_brake":
        EmergencyBrakeScenario,


    "cut_in_vehicle":
        CutInVehicleScenario,


    "pedestrian_crossing":
        PedestrianCrossingScenario,


    "static_obstacle":
        StaticObstacleScenario,

}



# =========================
# 打印场景
# =========================

def print_available_scenarios():

    print("\nAvailable scenarios:")

    for name in SCENARIOS:

        print(
            " -",
            name
        )



# =========================
# 输出结果
# =========================

def print_result(status):

    print(
        "\n========== Scenario Result =========="
    )


    print(
        "Status:",
        status.get(
            "status",
            "UNKNOWN"
        )
    )


    print(
        "Reason:",
        status.get(
            "reason",
            ""
        )
    )


    print(
        "Actors:",
        status.get(
            "actors",
            {}
        )
    )


    print(
        "Metrics:"
    )


    print(
        status.get(
            "metrics",
            {}
        )
    )


    print(
        "===================================="
    )



# =========================
# 主函数
# =========================

def main():


    # 参数检查

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
    # 创建Scenario
    # =====================

    scenario = SCENARIOS[scenario_name](
        world,
        external_control=True
    )



    try:

        print(
            "\n[Scenario]",
            scenario_name
        )


        # 初始化

        scenario.setup()


        print(
            scenario.get_scenario_info()
        )


        print(
            "[Scenario] Running..."
        )



        # =====================
        # 主循环
        # =====================

        while not scenario.finished():


            scenario.tick()


            world.tick()


            status = scenario.get_status()


            # 成功/失败退出

            if status["status"] != "RUNNING":

                break



            time.sleep(
                0.05
            )



    except KeyboardInterrupt:


        print(
            "\n[Scenario] Interrupted"
        )



    finally:


        # 输出最终状态

        try:

            print_result(
                scenario.get_status()
            )

        except Exception:

            pass



        print(
            "[Scenario] Cleaning..."
        )


        scenario.destroy()


        print(
            "[Scenario] Finished"
        )




if __name__ == "__main__":

    main()