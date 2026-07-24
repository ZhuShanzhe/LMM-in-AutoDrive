import carla

from perception.ground_truth import GroundTruth



class WorldState:


    def __init__(
        self,
        world,
        ego_vehicle
    ):


        self.world = world

        self.ego_vehicle = ego_vehicle


        # 使用已有真值模块
        self.ground_truth = GroundTruth(
            world,
            ego_vehicle
        )


        # 碰撞状态
        self.collision = False



    def update_collision(
        self,
        collision
    ):

        self.collision = collision



    def get_state(self):


        state = {


            # ==================
            # Ego状态
            # ==================

            "ego":
            self.ground_truth
            .get_state(),



            # ==================
            # 周围目标
            # ==================

            "vehicles":
            self.ground_truth
            .get_nearby_vehicles(),



            "pedestrians":
            self.ground_truth
            .get_nearby_pedestrians(),



            "obstacles":
            self.ground_truth
            .get_nearby_obstacles(),



            # ==================
            # 地图信息
            # ==================

            "lane":
            self.get_lane_info(),



            # ==================
            # 红绿灯
            # ==================

            "traffic_lights":
            self.get_traffic_lights(),



            # ==================
            # 天气
            # ==================

            "weather":
            self.get_weather(),



            # ==================
            # 碰撞
            # ==================

            "collision":
            {
                "status":
                self.collision
            }


        }


        return state





    # ======================
    # 车道信息
    # ======================

    def get_lane_info(self):


        location = (
            self.ego_vehicle
            .get_location()
        )


        waypoint = (
            self.world
            .get_map()
            .get_waypoint(
                location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
        )


        if waypoint is None:

            return None



        return {


            "road_id":
            waypoint.road_id,


            "section_id":
            waypoint.section_id,


            "lane_id":
            waypoint.lane_id,


            "lane_type":
            str(
                waypoint.lane_type
            ),


            "is_junction":
            waypoint.is_junction

        }




    # ======================
    # 红绿灯
    # ======================

    def get_traffic_lights(self):


        result = []


        lights = (
            self.world
            .get_actors()
            .filter(
                "traffic.traffic_light*"
            )
        )


        ego_location = (
            self.ego_vehicle
            .get_location()
        )


        for light in lights:


            distance = (
                ego_location
                .distance(
                    light.get_location()
                )
            )


            if distance < 80:


                result.append({

                    "id":
                    light.id,


                    "state":
                    str(
                        light.state
                    ),


                    "distance":
                    round(
                        distance,
                        2
                    )

                })


        return result




    # ======================
    # 天气
    # ======================

    def get_weather(self):


        weather = (
            self.world
            .get_weather()
        )


        return {


            "cloudiness":
            weather.cloudiness,


            "precipitation":
            weather.precipitation,


            "sun_altitude_angle":
            weather.sun_altitude_angle

        }