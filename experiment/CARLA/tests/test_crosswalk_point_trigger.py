from types import SimpleNamespace as NS
import pytest
from scenarios.complex.town05_scene2 import DeterministicSceneEvents, ScriptedWalker
import sys


def crossing_manager():
    event = dict(id='crosswalk_pedestrian', kind='crossing_pedestrian',
                 anchor_progress_m=932.4278, activate_progress_m=910.0417,
                 retain_after_completion=True,
                 ground_truth={'actor_roles': ['scene2_crosswalk_pedestrian']})
    clock=NS(now=0.)
    world=NS(get_actors=lambda:NS(filter=lambda pattern:[]),
        get_snapshot=lambda:NS(timestamp=NS(elapsed_seconds=clock.now)))
    manager = DeterministicSceneEvents(world, None, None, [], [], [event], 42)
    manager.test_clock=clock
    manager.states['bus_stop_passengers'] = 'STAGED'
    calls = []
    walker = NS(completed=False, start=lambda: calls.append('start'), update=lambda: None,
        activation_transform=NS(location=NS(x=0.,y=-5.,z=0.)),target=NS(x=0.,y=5.,z=0.))
    manager.scripted_walkers['crosswalk_pedestrian'] = walker
    manager.spawn_diagnostics['scene2_crosswalk_pedestrian'] = dict(
        trigger_location=dict(x=38.59, y=-155.42, z=0))
    return manager, walker, calls


def test_crossing_starts_at_point_once_despite_stops_or_progress_jitter():
    manager, walker, calls = crossing_manager()
    for progress in (0, 880, 910.0416):
        assert manager.update(progress) == []
    assert calls == []
    assert manager.update(910.0417)==[]
    manager.test_clock.now=.4
    assert manager.update(910.0417)==[] and calls==[]
    manager.test_clock.now=.6
    changes = manager.update(910.0417)
    assert calls == ['start']
    assert changes[0]['trigger_source'] == 'ego_route_progress'
    assert changes[0]['trigger_progress_m'] == 910.0417
    assert changes[0]['trigger_location']['x'] == 38.59
    for progress in (910.0417, 909.9, 910.1, 911):
        assert manager.update(progress) == []
    assert calls == ['start']
    assert manager.states['crosswalk_pedestrian'] == 'ACTIVE'
    walker.completed = True
    manager.update(911)
    assert manager.states['crosswalk_pedestrian'] == 'RESOLVED'
    manager.update(910)
    assert calls == ['start']


@pytest.mark.parametrize('entry', [910.0417, 911, 920])
def test_crossing_segment_start_past_point_still_activates_once(entry):
    manager, _, calls = crossing_manager()
    manager.update(entry)
    manager.test_clock.now=.6
    manager.update(entry + 1)
    assert calls == ['start']


def test_intermediate_crossing_endpoint_does_not_end_event(monkeypatch):
    monkeypatch.setitem(sys.modules,'carla',NS(WalkerControl=lambda **kw:NS(**kw),
                                              Vector3D=lambda **kw:NS(**kw)))
    position=NS(x=0,y=0)
    actor=NS(is_alive=True,get_location=lambda:position,apply_control=lambda control:None)
    final=NS(x=0,y=4)
    walker=ScriptedWalker(actor,NS(x=0,y=0),1.25,continuation_targets=[final])
    walker.start()
    walker.update()
    assert not walker.completed
    assert walker.target is final
    position.y=4
    walker.update()
    assert walker.completed
