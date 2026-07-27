import carla



class RoadFinder:


    def __init__(
        self,
        world
    ):

        self.world = world

        self.map = (
            world.get_map()
        )



    # =========================
    # 计算角度差
    # =========================

    def angle_difference(
        self,
        a,
        b
    ):

        diff = abs(a - b)


        if diff > 180:

            diff = 360 - diff


        return diff





    # =========================
    # 检查是否连续直路
    # =========================

    def check_straight_lane(
        self,
        waypoint,
        distance=100,
        step=5,
        direction="next"
    ):

        if direction not in ("next", "previous"):

            raise ValueError(

                "direction must be 'next' or 'previous'"

            )


        current = waypoint

        reference_yaw = waypoint.transform.rotation.yaw


        checked = 0



        while checked < distance:



            next_wps = getattr(current, direction)(step)



            # 没有道路

            if len(next_wps) == 0:

                return False



            # 分叉 / 路口

            if len(next_wps) > 1:

                return False



            next_wp = next_wps[0]



            yaw_diff = self.angle_difference(

                reference_yaw,

                next_wp.transform.rotation.yaw

            )



            # 弯道

            if yaw_diff > 10:

                return False



            current = next_wp


            checked += step



        return True





    # =========================
    # 寻找直道路段
    # =========================

    def find_straight_road(
        self,
        min_length=100,
        backward_length=0
    ):


        spawn_points = (

            self.map
            .get_spawn_points()

        )



        print(

            "[RoadFinder]"
            " Searching straight road..."

        )



        for idx, spawn in enumerate(
            spawn_points
        ):



            wp = (

                self.map
                .get_waypoint(

                    spawn.location,

                    project_to_road=True,

                    lane_type=carla.LaneType.Driving

                )

            )



            if wp is None:

                continue



            if self.check_straight_lane(

                wp,

                min_length

            ) and (

                backward_length <= 0

                or self.check_straight_lane(

                    wp,

                    backward_length,

                    direction="previous"

                )

            ):



                print(

                    "[RoadFinder]"
                    " Found road:",
                    idx

                )


                print(

                    "location:",
                    spawn.location

                )


                print(

                    "yaw:",
                    spawn.rotation.yaw

                )



                # 注意：
                # 返回官方spawn点
                # 不返回wp.transform

                return {


                    "spawn":

                        spawn,


                    "waypoint":

                        wp,


                    "length":

                        min_length


                }




        raise RuntimeError(

            "No straight road found"

        )
