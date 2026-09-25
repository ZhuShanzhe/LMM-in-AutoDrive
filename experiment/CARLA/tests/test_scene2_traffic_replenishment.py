"""Scene 2 ambient traffic must remain bounded and spawn outside the camera."""

import sys
from types import SimpleNamespace as NS

from scenarios.complex import town05_scene2 as scene


def test_replenishment_replaces_only_far_behind_ambient_actor(monkeypatch):
    locations = [NS(x=float(x), y=0.0, z=0.0) for x in range(0, 1001, 20)]
    waypoints = [NS(transform=NS(location=location, rotation=NS(yaw=0)),
                    is_junction=False, road_id=1, lane_id=-1, lane_type='Driving',
                    get_left_lane=lambda: None, get_right_lane=lambda: None)
                 for location in locations]
    route = [(waypoint, None) for waypoint in waypoints]
    deleted = []
    source = NS(id=1, is_alive=True, get_location=lambda: NS(x=-500.0, y=0.0),
                set_autopilot=lambda *args: None, destroy=lambda: deleted.append(1))
    replacement = NS(id=2, is_alive=True, set_autopilot=lambda *args: None)
    spawned = []
    world = NS(get_blueprint_library=lambda: None,
               try_spawn_actor=lambda blueprint, transform:
                   spawned.append(transform.location) or replacement)
    tm = NS(get_port=lambda: 8000, distance_to_leading_vehicle=lambda *args: None,
            vehicle_percentage_speed_difference=lambda *args: None,
            auto_lane_change=lambda *args: None,
            update_vehicle_lights=lambda *args: None)
    registry = scene.ActorRegistry()
    registry.add(source)
    flow = scene.TownTrafficFlow(None, world, tm, registry, route,
                                 dict(seed=1, vehicles=1, replenish_check_ticks=1,
                                      minimum_front_vehicles=4, replenish_lookahead_m=240))
    flow.vehicles = [source]
    monkeypatch.setitem(sys.modules, 'carla', NS(
        Location=lambda **kwargs: NS(**kwargs),
        Transform=lambda location, rotation: NS(location=location, rotation=rotation)))
    monkeypatch.setattr(scene, '_safe_car_blueprints', lambda library: [NS(has_attribute=lambda _: False)])
    ego = NS(get_location=lambda: NS(x=500.0, y=0.0),
             get_transform=lambda: NS(get_forward_vector=lambda: NS(x=1.0, y=0.0)))

    flow.maintain(ego, 500)

    assert len(spawned) == 1
    assert spawned[0].x >= 740
    assert deleted == [1]
    assert flow.vehicles == [replacement]
    assert registry.actors == [replacement]
    assert flow.replenishment_events[0]['spawn_distance_from_ego_m'] >= 240
    assert flow.replenishment_events[0]['retired_actor_id'] == 1
