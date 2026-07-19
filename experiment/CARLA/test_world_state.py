import carla

from perception.world_state import WorldState



def main():


    # =====================
    # 连接 CARLA
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
    # 获取一个已有车辆作为ego
    # =====================

    vehicles = (
        world
        .get_actors()
        .filter("vehicle.*")
    )


    if len(vehicles) == 0:


        print(
            "No vehicle found."
        )


        print(
            "Please spawn a vehicle first."
        )


        return



    ego_vehicle = vehicles[0]



    print(
        "Ego vehicle:",
        ego_vehicle.type_id
    )



    # =====================
    # 创建状态接口
    # =====================

    world_state = WorldState(
        world,
        ego_vehicle
    )



    # =====================
    # 获取状态
    # =====================

    state = (
        world_state
        .get_state()
    )



    print(
        "\n========== WORLD STATE =========="
    )


    print(state)



if __name__ == "__main__":

    main()