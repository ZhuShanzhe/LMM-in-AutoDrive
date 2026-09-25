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
