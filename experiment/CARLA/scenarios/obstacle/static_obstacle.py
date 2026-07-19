import carla


class StaticObstacleScenario:


    def __init__(self, world):

        self.world = world

        self.vehicle = None

        self.obstacle = None



    def setup(self):


        blueprint_library = (
            self.world
            .get_blueprint_library()
        )


        spawn_points = (
            self.world
            .get_map()
            .get_spawn_points()
        )


        # =====================
        # Ego Vehicle
        # =====================

        vehicle_bp = blueprint_library.find(
            "vehicle.tesla.model3"
        )


        self.vehicle = (
            self.world.spawn_actor(
                vehicle_bp,
                spawn_points[0]
            )
        )



        # =====================
        # Static obstacle
        # =====================


        obstacle_bp = (
            blueprint_library.find(
                "static.prop.streetbarrier"
            )
        )


        obstacle_transform = (
            spawn_points[1]
        )


        self.obstacle = (
            self.world.spawn_actor(
                obstacle_bp,
                obstacle_transform
            )
        )


        print(
            "[Scenario] Static obstacle created"
        )



    def destroy(self):


        if self.obstacle:

            self.obstacle.destroy()


        if self.vehicle:

            self.vehicle.destroy()


        print(
            "[Scenario] Clean"
        )