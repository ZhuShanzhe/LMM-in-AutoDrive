import carla

from scenarios.utils.road import RoadFinder



client = carla.Client(
    "localhost",
    2000
)

client.set_timeout(10)


world = client.get_world()



finder = RoadFinder(
    world
)



road = finder.find_straight_road()



print(road)