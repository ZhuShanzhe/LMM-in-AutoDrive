import carla
import json
import os
import math


class RouteManager:


    def __init__(self, world):

        self.world = world
        self.map = world.get_map()

        self.route = []

        self.route_length = 0

        self.progress = 0

        self.current_index = 0



    def build_route(
        self,
        start_location=None,
        length=8000,
        step=5
    ):


        if start_location:

            wp = self.map.get_waypoint(
                start_location,
                project_to_road=True
            )

        else:

            points = self.map.get_spawn_points()

            if not points:
                raise RuntimeError(
                    "No spawn points"
                )

            wp = self.map.get_waypoint(
                points[0].location
            )


        self.route=[]

        distance=0


        while distance < length:


            tf = wp.transform

            loc=tf.location


            self.route.append({

                "x":round(loc.x,3),
                "y":round(loc.y,3),
                "z":round(loc.z,3),

                "yaw":round(
                    tf.rotation.yaw,
                    3
                ),

                "distance":round(
                    distance,
                    3
                )

            })


            nxt=wp.next(step)


            if not nxt:
                break


            wp=nxt[0]

            distance += step



        self.route_length=distance


        print(
            "[Route]",
            round(distance,1),
            "m"
        )


        return self.route



    def save(self,path):

        folder=os.path.dirname(path)

        if folder:
            os.makedirs(
                folder,
                exist_ok=True
            )


        with open(
            path,
            "w",
            encoding="utf8"
        ) as f:

            json.dump(
                {
                    "map":self.map.name,
                    "length":self.route_length,
                    "waypoints":self.route
                },
                f,
                indent=2,
                ensure_ascii=False
            )



    def load(self,path):

        with open(
            path,
            encoding="utf8"
        ) as f:

            data=json.load(f)


        self.route=data["waypoints"]

        self.route_length=data["length"]



    def update(
        self,
        ego_vehicle
    ):


        if not self.route:
            return


        loc=(
            ego_vehicle
            .get_location()
        )


        best=float("inf")

        index=self.current_index



        start=max(
            0,
            index-50
        )

        end=min(
            len(self.route),
            index+200
        )


        for i in range(start,end):

            p=self.route[i]

            d=math.hypot(
                loc.x-p["x"],
                loc.y-p["y"]
            )


            if d < best:

                best=d

                index=i



        self.current_index=index


        self.progress=(
            self.route[index]["distance"]
        )



    def get_progress(self):

        return self.progress



    def get_remaining(self):

        return max(
            0,
            self.route_length-self.progress
        )


    def finished(self):

        return (
            self.progress >=
            self.route_length-10
        )