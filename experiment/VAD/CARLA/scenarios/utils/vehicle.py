import carla



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
    # 获取车辆蓝图
    # =========================

    def get_vehicle_blueprint(
        self,
        vehicle_type,
        role_name
    ):


        bp = (
            self.blueprint_library
            .find(
                vehicle_type
            )
        )


        if bp.has_attribute(
            "role_name"
        ):

            bp.set_attribute(
                "role_name",
                role_name
            )


        return bp





    # =========================
    # Ego车辆
    #
    # 不允许fallback
    # 场景必须确定
    # =========================

    def spawn_ego_vehicle(
        self,
        spawn_point,
        vehicle_type="vehicle.tesla.model3"
    ):


        bp = self.get_vehicle_blueprint(

            vehicle_type,

            "ego"

        )


        vehicle = (

            self.world
            .try_spawn_actor(

                bp,

                spawn_point

            )

        )


        if vehicle is None:


            raise RuntimeError(

                "Ego spawn failed. "
                "Clean CARLA world first."

            )


        return vehicle






    # =========================
    # NPC车辆
    # =========================

    def spawn_npc_vehicle(
        self,
        spawn_point,
        vehicle_type="vehicle.tesla.model3"
    ):


        bp = self.get_vehicle_blueprint(

            vehicle_type,

            "npc"

        )



        vehicle = (

            self.world
            .try_spawn_actor(

                bp,

                spawn_point

            )

        )


        if vehicle is None:


            raise RuntimeError(

                "NPC spawn failed"

            )


        return vehicle





    # =========================
    # 删除车辆
    # =========================

    def destroy_vehicle(
        self,
        vehicle
    ):


        if vehicle is not None:

            if vehicle.is_alive:

                vehicle.destroy()