import carla
import random


class RoadFinder:


    def __init__(self, world):

        self.world = world

        self.map = world.get_map()



    # =================================
    # 普通直道路
    # =================================

    def find_straight_road(
        self,
        min_length=100
    ):


        print(
            "[RoadFinder] Searching straight road..."
        )


        waypoints = self.map.generate_waypoints(
            2.0
        )


        candidates = []


        for wp in waypoints:


            length = 0

            current = wp


            for _ in range(
                int(min_length / 5)
            ):


                nxt = current.next(5)


                if len(nxt) == 0:
                    break


                current = nxt[0]

                length += 5



            if length >= min_length:


                candidates.append(wp)



        if len(candidates)==0:

            raise RuntimeError(
                "No straight road found"
            )



        wp = random.choice(
            candidates
        )



        print(
            "[RoadFinder] Found road:",
            wp.road_id
        )


        print(
            "location:",
            wp.transform.location
        )


        print(
            "yaw:",
            wp.transform.rotation.yaw
        )



        return {

            "waypoint": wp

        }





    # =================================
    # 多车道道路
    # 专用于:
    # cut-in
    # lane change
    # overtaking
    # =================================

    def find_multi_lane_road(
        self,
        min_length=100
    ):


        print(
            "[RoadFinder] Searching multi lane road..."
        )


        waypoints = self.map.generate_waypoints(
            2.0
        )


        candidates = []



        for wp in waypoints:



            # 必须是行驶车道

            if wp.lane_type != carla.LaneType.Driving:

                continue



            left = wp.get_left_lane()

            right = wp.get_right_lane()



            has_adjacent = False



            if left is not None:

                if left.lane_type == carla.LaneType.Driving:

                    has_adjacent = True



            if right is not None:

                if right.lane_type == carla.LaneType.Driving:

                    has_adjacent = True



            if not has_adjacent:

                continue



            # 检查长度

            length = 0

            current = wp


            for _ in range(
                int(min_length / 5)
            ):


                nxt = current.next(5)


                if len(nxt)==0:

                    break


                current = nxt[0]

                length += 5




            if length >= min_length:

                candidates.append(wp)




        if len(candidates)==0:


            raise RuntimeError(
                "No multi lane road found"
            )




        wp = random.choice(
            candidates
        )



        print(
            "[RoadFinder] Found multi lane road:",
            wp.road_id
        )


        print(
            "location:",
            wp.transform.location
        )


        print(
            "yaw:",
            wp.transform.rotation.yaw
        )



        return {

            "waypoint":wp

        }