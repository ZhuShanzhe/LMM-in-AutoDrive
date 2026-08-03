import numpy as np
import math


class GroundTruth:

    def __init__(self, world, vehicle):

        self.world = world
        self.vehicle = vehicle


    # ======================
    # Ego车辆状态
    # ======================

    def get_state(self):

        transform = (
            self.vehicle.get_transform()
        )

        velocity = (
            self.vehicle.get_velocity()
        )


        speed = (
            3.6 *
            np.sqrt(
                velocity.x ** 2 +
                velocity.y ** 2 +
                velocity.z ** 2
            )
        )


        return {

            "location": {

                "x": round(
                    transform.location.x, 2
                ),

                "y": round(
                    transform.location.y, 2
                ),

                "z": round(
                    transform.location.z, 2
                )

            },


            "rotation": {

                "pitch": round(
                    transform.rotation.pitch, 2
                ),

                "yaw": round(
                    transform.rotation.yaw, 2
                ),

                "roll": round(
                    transform.rotation.roll, 2
                )

            },


            # 修复 numpy.float64
            "speed(km/h)": float(
                round(speed, 2)
            )
        }



    # ======================
    # 周围车辆
    # ======================

    def get_nearby_vehicles(
            self,
            max_distance=50
    ):

        ego_transform = (
            self.vehicle.get_transform()
        )


        ego_location = (
            ego_transform.location
        )


        ego_yaw = math.radians(
            ego_transform.rotation.yaw
        )


        vehicles = (
            self.world
            .get_actors()
            .filter("vehicle.*")
        )


        result = []


        for vehicle in vehicles:


            if vehicle.id == self.vehicle.id:

                continue


            location = (
                vehicle.get_location()
            )


            distance = (
                ego_location.distance(
                    location
                )
            )


            if distance > max_distance:

                continue

            velocity = (
                vehicle.get_velocity()
            )

            speed_kmh = (
                3.6 *
                math.sqrt(
                    velocity.x ** 2 +
                    velocity.y ** 2 +
                    velocity.z ** 2
                )
            )



            dx = (
                location.x -
                ego_location.x
            )


            dy = (
                location.y -
                ego_location.y
            )


            relative_x = (
                dx * math.cos(ego_yaw)
                +
                dy * math.sin(ego_yaw)
            )


            relative_y = (
                -dx * math.sin(ego_yaw)
                +
                dy * math.cos(ego_yaw)
            )


            direction = (
                "front"
                if relative_x > 0
                else "rear"
            )


            result.append({

                "id": vehicle.id,

                "type": vehicle.type_id,

                "distance": round(
                    distance,
                    2
                ),

                "speed_kmh": float(
                    round(speed_kmh, 2)
                ),

                "relative_position": {

                    "x": round(
                        relative_x,
                        2
                    ),

                    "y": round(
                        relative_y,
                        2
                    )

                },

                "direction": direction

            })


        return result



    # ======================
    # 周围行人
    # ======================

    def get_nearby_pedestrians(
            self,
            max_distance=50
    ):

        ego_location = (
            self.vehicle.get_location()
        )


        pedestrians = (
            self.world
            .get_actors()
            .filter("walker.*")
        )


        result = []


        for pedestrian in pedestrians:


            distance = (
                ego_location.distance(
                    pedestrian.get_location()
                )
            )


            if distance <= max_distance:


                result.append({

                    "id": pedestrian.id,

                    "type": pedestrian.type_id,

                    "distance": round(
                        distance,
                        2
                    )

                })


        return result



    # ======================
    # 静态障碍物
    # ======================

    def get_nearby_obstacles(
            self,
            max_distance=50
    ):

        ego_location = (
            self.vehicle.get_location()
        )


        actors = (
            self.world
            .get_actors()
        )


        result = []


        for actor in actors:


            if (
                "static.prop"
                not in actor.type_id
            ):

                continue



            distance = (
                ego_location.distance(
                    actor.get_location()
                )
            )


            if distance <= max_distance:


                result.append({

                    "id": actor.id,

                    "type": actor.type_id,

                    "distance": round(
                        distance,
                        2
                    )

                })


        return result
