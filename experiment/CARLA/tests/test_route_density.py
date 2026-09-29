"""Exercise actor-pool maintenance without a simulator."""
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace


class Point:
    def __init__(self, x, y=0):
        self.x, self.y, self.z = x, y, 0

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def distance(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)


def make_pool(monkeypatch):
    monkeypatch.setitem(sys.modules, 'carla', SimpleNamespace())
    path = Path(__file__).parents[1] / 'scenarios/basic/urban_traffic.py'
    spec = importlib.util.spec_from_file_location('density_test_traffic', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    route = SimpleNamespace(route=[dict(x=x, y=0, distance_m=x) for x in range(0, 3001, 10)], route_length_m=3000)
    pool = module.FixedRouteTraffic(None, None, route, dict(density_lanes=[1, 2], density_ahead_end_m=550))
    pool._tick_count = 10
    pool._waypoint_at = lambda distance, lane: SimpleNamespace(
        is_junction=False, transform=SimpleNamespace(location=Point(distance, lane * 4),
            get_forward_vector=lambda: Point(1)))
    ego = SimpleNamespace(get_location=lambda: Point(500), get_velocity=lambda: Point(10))
    return pool, ego


def actor(identity, x):
    return SimpleNamespace(id=identity, is_alive=True, get_location=lambda: Point(x, 4))


def test_following_source_fills_lanes_without_rapid_respawn(monkeypatch):
    pool, ego = make_pool(monkeypatch)
    pool.route_manager.applied_directives = []
    pool.config.update(
        following_sources_enabled=True, following_source_check_ticks=10,
        following_source_back_m=100, following_source_spacing_m=65,
        density_lanes=[1, 2], density_max_actors=4,
        vehicles=[dict(id='template', vehicle_type='vehicle.audi.tt')],
    )
    pool._waypoint_at = lambda distance, lane: SimpleNamespace(
        is_junction=False, road_id=1, lane_id=lane,
        transform=SimpleNamespace(
            location=Point(distance, lane * 4),
            get_forward_vector=lambda: Point(1),
        ),
    )
    pool._same_direction_lanes = lambda waypoint: [
        pool._waypoint_at(waypoint.transform.location.x, lane) for lane in (1, 2)
    ]
    spawned = []

    def spawn(spec):
        spawned.append(spec)
        item = SimpleNamespace(
            id=len(spawned), is_alive=True,
            get_location=lambda: Point(spec['route_distance_m'], spec['lane_from_right'] * 4),
        )
        pool.actors.append(item)
        return item

    pool._spawn_vehicle = spawn
    pool._maintain_following_sources(ego, 500.0)
    pool._maintain_following_sources(ego, 500.0)
    pool._maintain_following_sources(ego, 500.0)
    assert len(spawned) == 2
    assert spawned[0]['lane_from_right'] != spawned[1]['lane_from_right']
    assert spawned[0]['speed_kmh'] == 55.0
    pool._tick_count += 10
    advanced_ego = SimpleNamespace(get_location=lambda: Point(570),
                                   get_velocity=lambda: Point(10))
    pool._maintain_following_sources(advanced_ego, 570.0)
    assert len(spawned) == 3


def test_initial_flow_thinning_rotates_across_all_four_lanes(monkeypatch):
    pool, _ = make_pool(monkeypatch)
    pool.route_manager.progress_m = 0.0
    pool.config.update(
        density_lanes=[1, 2, 3, 4], initial_spawn_divisor=2,
        vehicles=[dict(id=f'flow_{i}', lane_from_right=i % 4 + 1,
                       route_distance_m=50 + i * 10) for i in range(8)],
    )
    manager = SimpleNamespace(
        set_synchronous_mode=lambda value: None,
        set_random_device_seed=lambda value: None,
        set_global_distance_to_leading_vehicle=lambda value: None,
    )
    pool.client = SimpleNamespace(get_trafficmanager=lambda port: manager)
    spawned = []
    pool._spawn_vehicle = lambda spec: spawned.append(spec)

    pool.setup()

    assert [spec['id'] for spec in spawned] == [
        'flow_0', 'flow_2', 'flow_5', 'flow_7',
    ]
    assert {spec['lane_from_right'] for spec in spawned} == {1, 2, 3, 4}


def test_initial_flow_keeps_near_field_and_defers_distant_templates(monkeypatch):
    pool, _ = make_pool(monkeypatch)
    pool.route_manager.progress_m = 0.0
    pool.config.update(
        density_lanes=[1, 2, 3, 4], initial_spawn_divisor=2,
        initial_spawn_near_m=120, initial_spawn_max_m=420,
        vehicles=[dict(id=f'flow_{i}', lane_from_right=i % 4 + 1,
                       route_distance_m=50 + i * 50) for i in range(10)],
    )
    manager = SimpleNamespace(
        set_synchronous_mode=lambda value: None,
        set_random_device_seed=lambda value: None,
        set_global_distance_to_leading_vehicle=lambda value: None,
    )
    pool.client = SimpleNamespace(get_trafficmanager=lambda port: manager)
    spawned = []
    pool._spawn_vehicle = lambda spec: spawned.append(spec)

    pool.setup()

    assert [spec['route_distance_m'] for spec in spawned] == [50, 100, 150, 300, 400]


def test_resume_does_not_stack_traffic_at_route_end(monkeypatch):
    pool, _ = make_pool(monkeypatch)
    pool.route_manager.progress_m = 2500
    pool.config['vehicles'] = [
        dict(id='near', route_distance_m=100),
        dict(id='beyond', route_distance_m=1000),
    ]
    traffic_manager = SimpleNamespace(
        set_synchronous_mode=lambda value: None,
        set_random_device_seed=lambda value: None,
        set_global_distance_to_leading_vehicle=lambda value: None,
    )
    pool.client = SimpleNamespace(get_trafficmanager=lambda port: traffic_manager)
    spawned = []
    pool._spawn_vehicle = lambda spec: spawned.append(spec)
    pool.setup()
    assert len(spawned) == 1
    assert spawned[0]['route_distance_m'] == 2600


def test_actor_probe_reports_route_alignment_without_policy_input(monkeypatch):
    pool, _ = make_pool(monkeypatch)
    pool.route_manager.route = [dict(x=x, y=0, distance_m=x) for x in range(0, 301, 5)]
    pool.world = SimpleNamespace(get_map=lambda: SimpleNamespace(
        get_waypoint=lambda location, **kwargs:
            SimpleNamespace(road_id=7, lane_id=2)))
    pool.actors = [SimpleNamespace(
        id=4, is_alive=True, get_location=lambda: Point(100, 3),
        get_velocity=lambda: Point(8), get_speed_limit=lambda: 30.0,
    )]
    pool._desired_speeds[4] = 28.0
    pool.config['enabled'] = True
    ego = SimpleNamespace(
        get_location=lambda: Point(0),
        get_transform=lambda: SimpleNamespace(
            get_forward_vector=lambda: Point(1)),
    )
    sys.modules['carla'].LaneType = SimpleNamespace(Driving=1)

    result = pool.actor_probe(ego)

    assert result['scope'] == 'simulator_audit_only'
    assert result['actors'][0]['nearest_route_s_m'] == 100.0
    assert result['actors'][0]['along_ego_m'] == 100.0
    assert result['actors'][0]['desired_speed_kmh'] == 28.0


def test_other_road_actor_does_not_occupy_route_density_cell(monkeypatch):
    pool, ego = make_pool(monkeypatch)
    pool.config.update(
        density_lanes=[1], density_ahead_start_m=180,
        density_ahead_end_m=200, density_spacing_m=90,
        lifecycle_protected_radius_m=150, density_max_actors=2,
        vehicles=[dict(id='template', route_distance_m=100)],
    )
    pool.actors = [actor(1, 700)]
    pool._waypoint_at = lambda distance, lane: SimpleNamespace(
        road_id=1, lane_id=1, is_junction=False,
        transform=SimpleNamespace(location=Point(distance, 4),
                                  get_forward_vector=lambda: Point(1)),
    )
    pool.world = SimpleNamespace(get_map=lambda: SimpleNamespace(
        get_waypoint=lambda location, **kwargs:
            SimpleNamespace(road_id=2, lane_id=1)))
    sys.modules['carla'].LaneType = SimpleNamespace(Driving=1)
    spawned = []
    pool._spawn_vehicle = lambda spec: spawned.append(spec) or actor(2, 680)

    pool._maintain_route_density(ego, 500)

    assert len(spawned) == 1
    assert spawned[0]['route_distance_m'] == 680


def test_density_can_extend_beyond_ego_route_end(monkeypatch):
    pool, ego = make_pool(monkeypatch)
    pool.config.update(density_extend_past_goal_m=220, density_ahead_start_m=175,
                       density_ahead_end_m=425, lifecycle_protected_radius_m=150)
    pool.actors = [actor(1, 0)]
    calls = []
    pool._replace_actor = lambda source, distance, lane, waypoint, **kwargs: (
        calls.append(distance) or actor(source.id + 100, distance)
    )
    pool._maintain_route_density(ego, 2900)
    assert calls and all(distance > pool.route_manager.route_length_m for distance in calls)


def test_protect_nearby_and_upcoming_actors(monkeypatch):
    pool, ego = make_pool(monkeypatch)
    pool.actors = [actor(1, 450), actor(2, 950)]
    pool._replace_actor = lambda *a, **k: (_ for _ in ()).throw(AssertionError('protected actor recycled'))
    pool._maintain_route_density(ego, 500)
    assert pool._recycle_count == 0
    assert pool._density_diagnostics['gaps'] > 0


def test_recycle_behind_into_bounded_ahead_pool(monkeypatch):
    pool, ego = make_pool(monkeypatch)
    pool.actors = [actor(i, 0) for i in range(5)]
    calls = []

    def replace(source, distance, lane, waypoint, **kwargs):
        calls.append((source.id, distance, lane))
        return actor(source.id + 100, distance)

    pool._replace_actor = replace
    pool._maintain_route_density(ego, 500)
    assert len(calls) == 2
    assert all(900 <= c[1] <= 1050 for c in calls)
    assert pool._recycle_count == 2


def test_junctions_are_not_spawn_sites(monkeypatch):
    pool, ego = make_pool(monkeypatch)
    pool.actors = [actor(1, 0)]
    pool._waypoint_at = lambda *args: SimpleNamespace(is_junction=True)
    pool._replace_actor = lambda *a, **k: (_ for _ in ()).throw(AssertionError('junction spawn'))
    pool._maintain_route_density(ego, 500)
    assert pool._recycle_count == 0


def test_recent_replacement_cannot_be_recycled_again(monkeypatch):
    pool, ego = make_pool(monkeypatch)
    pool.actors = [actor(1, 0)]
    pool._last_actor_recycle_tick[1] = 5
    pool._replace_actor = lambda *a, **k: (_ for _ in ()).throw(AssertionError('cooldown ignored'))
    pool._maintain_route_density(ego, 500)
    assert pool._recycle_count == 0


def test_reserve_growth_is_bounded_and_does_not_move_nearby_cars(monkeypatch):
    pool, ego = make_pool(monkeypatch)
    pool.actors = [actor(1, 450)]
    pool.config.update(density_max_actors=2, vehicles=[dict(id='template')])
    calls = []

    def spawn(spec):
        calls.append(spec)
        created = actor(100, spec['route_distance_m'])
        pool.actors.append(created)
        return created

    pool._spawn_vehicle = spawn
    pool._maintain_route_density(ego, 500)
    assert len(calls) == 1
    assert calls[0]['route_distance_m'] >= 900
    assert len(pool.actors) == 2
    assert pool._density_diagnostics['added'] == 1
    assert pool._recycle_count == 0


def test_scripted_actor_is_not_recycled(monkeypatch):
    pool, ego = make_pool(monkeypatch)
    special = actor(1, 0)
    pool.actors = [special]
    pool._scripted = [dict(actor=special)]
    pool._replace_actor = lambda *a, **k: (_ for _ in ()).throw(AssertionError('script actor recycled'))
    pool._maintain_route_density(ego, 500)
    assert pool._recycle_count == 0


def test_scripted_vehicle_lane_keeps_a_clear_approach(monkeypatch):
    pool, ego = make_pool(monkeypatch)
    pool.config.update(density_lanes=[1], density_ahead_start_m=400,
                       density_ahead_end_m=620, density_spacing_m=90,
                       density_scripted_clearance_m=120)
    source = actor(1, 0)
    scripted = actor(2, 900)
    pool.actors = [source, scripted]
    pool._scripted = [dict(actor=scripted)]
    targets = []
    pool._replace_actor = lambda _actor, distance, _lane, _waypoint, **_kwargs: (
        targets.append(distance) or actor(3, distance)
    )

    pool._maintain_route_density(ego, 500)

    assert targets == [1080]


def test_density_respects_explicit_event_clear_window(monkeypatch):
    pool, ego = make_pool(monkeypatch)
    pool.config.update(density_lanes=[1], density_ahead_start_m=400,
                       density_ahead_end_m=620, density_spacing_m=90,
                       traffic_clear_windows_m=[[880, 1000]])
    pool.actors = [actor(1, 0)]
    targets = []
    pool._replace_actor = lambda _actor, distance, _lane, _waypoint, **_kwargs: (
        targets.append(distance) or actor(3, distance)
    )

    pool._maintain_route_density(ego, 500)

    assert targets == [1080]


def test_failed_reserve_spawn_does_not_consume_capacity(monkeypatch):
    pool, ego = make_pool(monkeypatch)
    pool.config.update(density_max_actors=2, vehicles=[dict(id='template')])
    pool._spawn_vehicle = lambda spec: None
    pool._maintain_route_density(ego, 500)
    assert pool._density_spawn_count == 0
    assert pool._density_diagnostics['added'] == 0


def test_absolute_speed_mode_preserves_scripted_slowdown(monkeypatch):
    pool, ego = make_pool(monkeypatch)
    special = actor(1, 0)
    special.get_speed_limit = lambda: 60
    pool.config['absolute_cruise_speed'] = True
    pool._scripted = [dict(actor=special, triggered=False, trigger_at_progress_m=400,
                           speed_kmh=15, id='slow_car')]
    speeds = []
    pool.traffic_manager = SimpleNamespace(
        vehicle_percentage_speed_difference=lambda *args: None,
        set_desired_speed=lambda a, speed: speeds.append(speed))
    pool._maintain_visible_pool = lambda *args: None
    pool.tick(ego, 500)
    pool.tick(ego, 600)
    assert speeds == [15]
