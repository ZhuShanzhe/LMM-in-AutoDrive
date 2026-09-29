"""A late cyclist must not block an earlier pass of the repeated route."""

import sys
from types import SimpleNamespace as NS

from scenarios.complex import town05_scene2 as scene


def test_cyclist_is_placed_only_after_route_overlap_and_out_of_view(monkeypatch):
    placed = []
    physics = []
    cyclist = NS(id=7, is_alive=True,
                 set_transform=lambda transform: placed.append(transform),
                 set_simulate_physics=lambda enabled: physics.append(enabled),
                 apply_control=lambda control: None)
    ego_position = NS(x=292.0, y=0.0)
    ego = NS(get_location=lambda: ego_position)
    world = NS(get_actors=lambda: NS(filter=lambda kind: []))
    events = scene.DeterministicSceneEvents(
        world, None, scene.ActorRegistry(), [], [],
        [dict(id='slow_cyclist', kind='cyclist', anchor_progress_m=1575,
              activate_progress_m=1500)], seed=1, ego=ego,
    )
    events.cyclist = cyclist
    events.spawn_diagnostics['scene2_slow_cyclist'] = {}
    events._waypoint = lambda progress: NS(transform=NS(
        location=NS(x=300.0, y=0.0, z=0.0), rotation=NS(yaw=0)))
    monkeypatch.setitem(sys.modules, 'carla', NS(
        Location=lambda **kwargs: NS(**kwargs),
        Transform=lambda location, rotation: NS(location=location, rotation=rotation),
        VehicleControl=lambda **kwargs: NS(**kwargs),
    ))
    event = events.events[0]

    events._prestage_cyclist(event, 343)
    assert placed == []

    ego_position.x = 0.0
    events._prestage_cyclist(event, 1050)
    assert len(placed) == 1
    assert placed[0].location.x == 300.0
    assert physics == [True]
    assert events.spawn_diagnostics['scene2_slow_cyclist']['prestage_distance_from_ego_m'] == 300.0
