"""Scene 2 ambient traffic must remain bounded and spawn outside the camera."""

import sys
from types import SimpleNamespace as NS

from scenarios.complex import town05_scene2 as scene


def test_replenishment_replaces_only_far_behind_ambient_actor(monkeypatch):
    locations = [NS(x=float(x), y=0.0, z=0.0) for x in range(0, 1001, 20)]
    waypoints = [NS(transform=NS(location=location, rotation=NS(yaw=0),
                                get_forward_vector=lambda: NS(x=1.0, y=0.0)),
                    is_junction=False, road_id=1, lane_id=-1, lane_type='Driving',
                    get_left_lane=lambda: None, get_right_lane=lambda: None)
                 for location in locations]
    route = [(waypoint, None) for waypoint in waypoints]
    deleted = []
    source = NS(id=1, is_alive=True, get_location=lambda: NS(x=-500.0, y=0.0),
                get_transform=lambda: NS(get_forward_vector=lambda: NS(x=1.0, y=0.0)),
                set_autopilot=lambda *args: None, destroy=lambda: deleted.append(1))
    replacement = NS(id=2, is_alive=True, set_autopilot=lambda *args: None)
    spawned = []
    world = NS(get_blueprint_library=lambda: None,
               try_spawn_actor=lambda blueprint, transform:
                   spawned.append(transform.location) or replacement)
    paths = []
    speeds = []
    tm = NS(get_port=lambda: 8000, set_path=lambda actor, path: paths.append(path),
            distance_to_leading_vehicle=lambda *args: None,
            vehicle_percentage_speed_difference=lambda *args: None,
            set_desired_speed=lambda actor, speed: speeds.append(speed),
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
    assert spawned[0].x >= 860
    assert deleted == [1]
    assert flow.vehicles == [replacement]
    assert registry.actors == [replacement]
    assert flow.replenishment_events[0]['spawn_distance_from_ego_m'] >= 350
    assert flow.replenishment_settings['visibility_clearance_m'] == 350
    assert flow.replenishment_events[0]['retired_actor_id'] == 1
    assert paths and paths[0][0].x > spawned[0].x
    assert len(speeds) == 1 and 28.0 <= speeds[0] <= 42.0


def test_initial_route_traffic_cycles_three_lane_preferences(monkeypatch):
    waypoint = NS(transform=NS(location=NS(x=0.0, y=0.0, z=0.0)))
    flow = scene.TownTrafficFlow(
        None, None, None, scene.ActorRegistry(), [(waypoint, None)],
        dict(seed=1, vehicles=0),
    )
    choices = []
    monkeypatch.setattr(flow, '_spawn_vehicles', lambda *args: None)
    monkeypatch.setattr(flow, '_spawn_walkers', lambda: None)
    monkeypatch.setattr(
        flow, '_spawn_route_vehicle',
        lambda *args, **kwargs: choices.append(kwargs['lane_choice']) or None,
    )

    flow.spawn([], waypoint.transform.location)

    assert choices == [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]


def test_opposite_direction_traffic_does_not_satisfy_same_direction_gate():
    def vehicle(x, direction):
        return NS(is_alive=True, get_location=lambda: NS(x=x, y=0.0),
                  get_transform=lambda: NS(get_forward_vector=lambda: NS(
                      x=direction, y=0.0)))

    ego = vehicle(0.0, 1.0)
    front, same_visible, same_nearby = scene.TownTrafficFlow._nearby_traffic_counts(
        ego, [vehicle(40.0, -1.0), vehicle(60.0, -1.0),
              vehicle(80.0, -1.0), vehicle(100.0, -1.0),
              vehicle(180.0, 1.0)]
    )

    assert (front, same_visible, same_nearby) == (4, 0, 1)


def test_route_count_excludes_parallel_road_and_opposite_direction():
    points = [NS(x=float(x), y=0.0, z=0.0) for x in range(0, 601, 5)]
    waypoints = [NS(transform=NS(location=point, get_forward_vector=lambda: NS(x=1.0, y=0.0)))
                 for point in points]
    route = [(waypoint, None) for waypoint in waypoints]

    def vehicle(x, y, direction=1.0):
        return NS(is_alive=True, get_location=lambda: NS(x=x, y=y),
                  get_transform=lambda: NS(get_forward_vector=lambda: NS(x=direction, y=0.0)))

    flow = scene.TownTrafficFlow(None, None, None, scene.ActorRegistry(), route,
                                 dict(seed=1, vehicles=0))
    flow.vehicles = [vehicle(50.0, 0.0), vehicle(180.0, 3.5),
                     vehicle(420.0, 0.0),
                     vehicle(80.0, 25.0), vehicle(90.0, 0.0, -1.0)]

    assert flow._route_ahead_counts(0.0) == (1, 3)


def test_route_count_rejects_nearby_different_road():
    points = [NS(x=float(x), y=0.0, z=0.0) for x in range(0, 601, 5)]
    waypoints = [NS(road_id=1, section_id=0, lane_id=-1,
                    transform=NS(location=point,
                                 get_forward_vector=lambda: NS(x=1.0, y=0.0)))
                 for point in points]
    road_map = NS(get_waypoint=lambda location, **_kwargs: NS(
        road_id=location.road_id, section_id=0, lane_id=-1))
    flow = scene.TownTrafficFlow(
        None, NS(get_map=lambda: road_map), None, scene.ActorRegistry(),
        [(waypoint, None) for waypoint in waypoints], dict(seed=1, vehicles=0),
    )

    def vehicle(x, road_id):
        return NS(is_alive=True,
                  get_location=lambda: NS(x=x, y=3.5, road_id=road_id),
                  get_transform=lambda: NS(get_forward_vector=lambda: NS(x=1.0, y=0.0)))

    flow.vehicles = [vehicle(50.0, 1), vehicle(80.0, 2)]
    assert flow._route_ahead_counts(0.0) == (1, 1)


def test_replenished_vehicles_use_spaced_out_route_anchors(monkeypatch):
    points = [NS(x=float(x), y=0.0, z=0.0) for x in range(0, 1001, 20)]
    route = [(NS(road_id=1, lane_id=-1, lane_type='Driving', is_junction=False,
                 get_left_lane=lambda: None, get_right_lane=lambda: None,
                 transform=NS(location=point, get_forward_vector=lambda: NS(x=1.0, y=0.0))), None)
             for point in points]
    flow = scene.TownTrafficFlow(
        None, None, NS(get_port=lambda: 8000), scene.ActorRegistry(), route,
        dict(seed=1, vehicles=0, replenish_check_ticks=1),
    )
    ego = NS(get_location=lambda: NS(x=100.0, y=0.0),
             get_transform=lambda: NS(get_forward_vector=lambda: NS(x=1.0, y=0.0)))
    starts = []

    def spawn(_origin, route_s_m, **_kwargs):
        starts.append(route_s_m)
        location = NS(x=route_s_m, y=0.0)
        actor = NS(id=len(starts), is_alive=True, get_location=lambda: location)
        return actor, location

    monkeypatch.setattr(flow, '_nearby_traffic_counts', lambda *args: (0, 0, 0))
    monkeypatch.setattr(flow, '_route_ahead_counts', lambda *args: (0, 0))
    monkeypatch.setattr(flow, '_spawn_route_vehicle', spawn)

    flow.maintain(ego, 100.0)
    flow.maintain(ego, 100.0)

    assert starts == [460.0, 495.0]
    assert [event['requested_spawn_route_m'] for event in flow.replenishment_events] == starts


def test_aggregate_count_does_not_mask_an_empty_route_lane(monkeypatch):
    points = [NS(x=float(x), y=0.0, z=0.0) for x in range(0, 1001, 20)]
    route = [(NS(transform=NS(location=point)), None) for point in points]
    flow = scene.TownTrafficFlow(
        None, None, NS(get_port=lambda: 8000), scene.ActorRegistry(), route,
        dict(seed=1, vehicles=0, replenish_check_ticks=1,
             minimum_front_vehicles=4),
    )
    ego = NS(get_location=lambda: NS(x=100.0, y=0.0),
             get_transform=lambda: NS(get_forward_vector=lambda: NS(x=1.0, y=0.0)))
    spawned = []
    monkeypatch.setattr(flow, '_nearby_traffic_counts', lambda *args: (0, 0, 0))
    monkeypatch.setattr(flow, '_staging_lane_gaps', lambda *args: [(460.0, 0)])
    monkeypatch.setattr(flow, '_spawn_route_vehicle', lambda *args, **kwargs:
                        spawned.append(args[1]) or (NS(id=3), NS(x=460.0, y=0.0)))

    monkeypatch.setattr(flow, '_route_ahead_counts', lambda *args: (0, 5))
    flow.maintain(ego, 100.0)
    assert spawned == [460.0]

    monkeypatch.setattr(flow, '_route_ahead_counts', lambda *args: (2, 5))
    flow.maintain(ego, 100.0)
    assert spawned == [460.0, 460.0]


def test_staging_prefers_lane_with_fewer_vehicles():
    def lane_group(x):
        lanes = [NS(road_id=1, section_id=0, lane_id=-(rank + 1),
                    lane_type='Driving', is_junction=False,
                    transform=NS(location=NS(x=float(x), y=4.0 * rank),
                                 get_forward_vector=lambda: NS(x=1.0, y=0.0)))
                 for rank in range(3)]
        for rank, lane in enumerate(lanes):
            lane.get_right_lane = lambda rank=rank, lanes=lanes: lanes[rank - 1] if rank else None
            lane.get_left_lane = lambda rank=rank, lanes=lanes: lanes[rank + 1] if rank < 2 else None
        return lanes[1]

    route = [(lane_group(x), None) for x in range(0, 1001, 20)]
    flow = scene.TownTrafficFlow(
        None, None, None, scene.ActorRegistry(), route,
        dict(seed=1, vehicles=0, replenish_staging_window_m=150),
    )
    flow.vehicles = [
        NS(is_alive=True, get_location=lambda y=y: NS(x=380.0, y=y))
        for y in (0.0, 4.0)
    ]

    gaps = flow._staging_lane_gaps(NS(x=0.0, y=0.0), 0.0)

    assert gaps
    assert gaps[0][1] == 2


def test_staging_excludes_junctions_and_reserved_event_positions():
    route = []
    for x in range(0, 1001, 20):
        waypoint = NS(road_id=1, lane_id=-1, lane_type='Driving',
                      is_junction=x == 360,
                      get_right_lane=lambda: None, get_left_lane=lambda: None,
                      transform=NS(location=NS(x=float(x), y=0.0),
                                   get_forward_vector=lambda: NS(x=1.0, y=0.0)))
        route.append((waypoint, None))
    flow = scene.TownTrafficFlow(
        None, None, None, scene.ActorRegistry(), route,
        dict(seed=1, vehicles=0, replenish_staging_window_m=100),
    )
    flow.reserved_locations = [NS(x=440.0, y=0.0)]

    assert flow._staging_lane_gaps(NS(x=0.0, y=0.0), 0.0) == []


def test_destroyed_background_actor_releases_pool_capacity(monkeypatch):
    route = [(NS(transform=NS(location=NS(x=float(x), y=0.0))), None)
             for x in range(0, 1001, 20)]
    registry = scene.ActorRegistry()
    dead = NS(id=1, is_alive=False)
    registry.add(dead)
    flow = scene.TownTrafficFlow(
        None, None, NS(get_port=lambda: 8000), registry, route,
        dict(seed=1, vehicles=1, maximum_extra_actors=0, replenish_check_ticks=1),
    )
    flow.vehicles = [dead]
    replacement = NS(id=2, is_alive=True)
    monkeypatch.setattr(flow, '_nearby_traffic_counts', lambda *args: (0, 0, 0))
    monkeypatch.setattr(flow, '_route_ahead_counts', lambda *args: (0, 0))
    monkeypatch.setattr(flow, '_staging_lane_gaps', lambda *args: [(460.0, 0)])
    monkeypatch.setattr(flow, '_spawn_route_vehicle', lambda *args, **kwargs:
                        (replacement, NS(x=460.0, y=0.0)))
    ego = NS(get_location=lambda: NS(x=100.0, y=0.0))

    flow.maintain(ego, 100.0)

    assert dead not in flow.vehicles
    assert dead not in registry.actors
    assert flow.replenishment_events[0]['new_actor_id'] == replacement.id


def _three_lane_source_route():
    route = []
    for x in range(0, 1001, 20):
        lanes = [NS(road_id=1, lane_id=-(rank + 1), lane_type='Driving',
                    is_junction=False,
                    transform=NS(location=NS(x=float(x), y=rank * 4.0),
                                 get_forward_vector=lambda: NS(x=1.0, y=0.0)))
                 for rank in range(3)]
        for rank, lane in enumerate(lanes):
            lane.get_right_lane = lambda rank=rank, lanes=lanes: lanes[rank - 1] if rank else None
            lane.get_left_lane = lambda rank=rank, lanes=lanes: lanes[rank + 1] if rank < 2 else None
        route.append((lanes[1], None))
    return route


def test_following_lane_sources_move_with_ego_and_enforce_spacing():
    flow = scene.TownTrafficFlow(
        None, None, None, scene.ActorRegistry(), _three_lane_source_route(),
        dict(seed=1, vehicles=0),
    )

    first = flow._following_lane_sources(NS(x=500.0, y=4.0), 500.0)
    assert len(first) == 3
    assert first[0][0] == 400.0
    source = flow._following_sources[first[0][2]]
    source.last_spawn_progress_m = 500.0

    second = flow._following_lane_sources(NS(x=520.0, y=4.0), 520.0)
    assert all(key != first[0][2] for _, _, key in second)
    assert source.route_s_m == 420.0

    third = flow._following_lane_sources(NS(x=580.0, y=4.0), 580.0)
    assert any(key == first[0][2] for _, _, key in third)
    assert source.route_s_m == 480.0


def test_following_lane_sources_skip_occupied_lane_and_junction():
    route = _three_lane_source_route()
    flow = scene.TownTrafficFlow(
        None, None, None, scene.ActorRegistry(), route,
        dict(seed=1, vehicles=0),
    )
    flow.vehicles = [NS(is_alive=True, get_location=lambda: NS(x=405.0, y=0.0))]
    candidates = flow._following_lane_sources(NS(x=500.0, y=4.0), 500.0)
    assert {rank for _, rank, _ in candidates} == {1, 2}

    route[25][0].is_junction = True
    assert flow._following_lane_sources(NS(x=500.0, y=4.0), 500.0) == []


def test_following_lane_source_spawn_is_recorded(monkeypatch):
    flow = scene.TownTrafficFlow(
        None, None, NS(get_port=lambda: 8000), scene.ActorRegistry(),
        _three_lane_source_route(),
        dict(seed=1, vehicles=0, replenish_check_ticks=1),
    )
    flow._replenished = 2
    ego = NS(get_location=lambda: NS(x=500.0, y=4.0),
             get_transform=lambda: NS(get_forward_vector=lambda: NS(x=1.0, y=0.0)))
    calls = []
    replacement = NS(id=42, is_alive=True)
    monkeypatch.setattr(flow, '_nearby_traffic_counts', lambda *args: (0, 0, 0))
    monkeypatch.setattr(flow, '_route_ahead_counts', lambda *args: (0, 0))
    monkeypatch.setattr(flow, '_staging_lane_gaps', lambda *args: [(860.0, 0)])
    monkeypatch.setattr(flow, '_spawn_route_vehicle', lambda *args, **kwargs:
                        calls.append((args[1], kwargs)) or
                        (replacement, NS(x=args[1], y=0.0)))

    flow.maintain(ego, 500.0)

    assert calls[0][0] == 400.0
    assert calls[0][1]['speed_range_kmh'] == (50.0, 60.0)
    assert flow.replenishment_events[0]['source_kind'] == 'following_lane_source'
    assert any(replacement in source.actors for source in flow._following_sources.values())


def test_distant_ambient_actor_frees_a_slot_for_following_source(monkeypatch):
    registry = scene.ActorRegistry()
    removed = []
    distant = NS(id=1, is_alive=True, get_location=lambda: NS(x=1100.0, y=0.0),
                 set_autopilot=lambda *args: None,
                 destroy=lambda: removed.append(1))
    registry.add(distant)
    flow = scene.TownTrafficFlow(
        None, None, NS(get_port=lambda: 8000), registry,
        _three_lane_source_route(),
        dict(seed=1, vehicles=1, maximum_extra_actors=0,
             replenish_check_ticks=1),
    )
    flow.vehicles = [distant]
    flow._replenished = 2
    ego = NS(get_location=lambda: NS(x=500.0, y=4.0),
             get_transform=lambda: NS(get_forward_vector=lambda: NS(x=1.0, y=0.0)))
    replacement = NS(id=2, is_alive=True)
    monkeypatch.setattr(flow, '_nearby_traffic_counts', lambda *args: (0, 0, 0))
    monkeypatch.setattr(flow, '_route_ahead_counts', lambda *args: (0, 0))
    monkeypatch.setattr(flow, '_staging_lane_gaps', lambda *args: [])
    monkeypatch.setattr(flow, '_spawn_route_vehicle', lambda *args, **kwargs:
                        (replacement, NS(x=400.0, y=0.0)))

    flow.maintain(ego, 500.0)

    assert removed == [1]
    assert flow.vehicles == []  # The spawn stub does not append its replacement.
    assert flow.replenishment_events[0]['retired_actor_id'] == 1
