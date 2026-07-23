import carla

from Route_manager import RouteManager


client=carla.Client(
    "localhost",
    2000
)

client.set_timeout(10)

world=client.get_world()


route_manager=RouteManager(
    world
)


route_manager.build_route(
    length=8000
)


route_manager.save(
    "route/routes/town10_route.json"
)