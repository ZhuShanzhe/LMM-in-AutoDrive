import carla
import time


from ScenarioManager import ScenarioManager


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
# Scenario注册
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
        StaticObstacleScenario

}




# =========================
# 获取ego
# =========================

def get_or_spawn_ego(world):


    vehicles = (
        world
        .get_actors()
        .filter(
            "vehicle.*"
        )
    )


    # 优先寻找外部ego

    for v in vehicles:


        if (
            v.attributes.get(
                "role_name",
                ""
            )
            ==
            "ego"
        ):

            print(
                "[LongScene] Use existing ego:",
                v.id
            )

            return v



    # =====================
    # 测试模式生成ego
    # =====================


    print(
        "[LongScene] No ego found, spawn test ego"
    )


    bp = (
        world
        .get_blueprint_library()
        .find(
            "vehicle.tesla.model3"
        )
    )


    bp.set_attribute(
        "role_name",
        "ego"
    )


    spawn_points = (
        world
        .get_map()
        .get_spawn_points()
    )


    if not spawn_points:

        raise RuntimeError(
            "No spawn points"
        )


    ego = (
        world
        .try_spawn_actor(
            bp,
            spawn_points[0]
        )
    )


    if ego is None:

        raise RuntimeError(
            "Spawn ego failed"
        )


    print(
        "[LongScene] Test ego:",
        ego.id
    )


    return ego





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


    world = (
        client
        .get_world()
    )



    # =====================
    # ScenarioManager
    # =====================


    manager = ScenarioManager(
        world,
        fixed_delta_s=0.05
    )



    for name, cls in SCENARIOS.items():

        manager.register(
            name,
            cls
        )



    # =====================
    # ego
    # =====================


    ego_vehicle = get_or_spawn_ego(
        world
    )



    # =====================
    # 加载8km配置
    # =====================


    manager.load_long_scene(

        "configs/town10_8km.json"

    )



    print(
        "[LongScene] Start"
    )



    try:


        while True:


            # 更新事件

            manager.tick_long_scene(
                ego_vehicle
            )



            # 推进CARLA

            world.tick()



            # 路线结束

            if (
                manager
                .route_manager
                .finished()
            ):

                print(
                    "[LongScene] Route finished"
                )

                break



            time.sleep(
                0.05
            )



    except KeyboardInterrupt:


        print(
            "[LongScene] Interrupted"
        )



    finally:


        print(
            "[LongScene] Cleaning"
        )


        manager.destroy()



        print(
            "[LongScene] Done"
        )





if __name__ == "__main__":

    main()