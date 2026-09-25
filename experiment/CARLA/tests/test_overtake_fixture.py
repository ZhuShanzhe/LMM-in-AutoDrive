import copy
import json
import sys
from types import SimpleNamespace as NS

import pytest
from benchmark.catalog import CONFIG_ROOT,ConfigError
from benchmark.overtake_fixture import OvertakeFixture


def prepare(monkeypatch):
    monkeypatch.setitem(sys.modules,'carla',NS(Location=NS))
    pair=dict(direction='LEFT',legal=True,entry_lane_key='1:0:-1',target_lane_key='1:0:-2')
    route=[dict(x=x,y=0,z=0,distance_m=x) for x in range(0,121,5)]
    geometry=dict(route=route,source_entry_m=1407,outgoing=pair,
                  lane_pairs=[dict(distance_m=0,outgoing=pair)])
    monkeypatch.setattr('benchmark.overtake_fixture.build_overtake_geometry',lambda *a,**kw:geometry)
    spec=json.loads((CONFIG_ROOT/'benchmark/s2_t05_cmd_07.json').read_text())
    events=[dict(kind='cyclist',target_speed_kmh=14,
                 ground_truth={'actor_roles':['scene2_slow_cyclist']})]
    return NS(get_map=lambda:NS()),spec,events


def test_isolation_rebases_without_mutating_source(monkeypatch):
    world,spec,events=prepare(monkeypatch)
    original=copy.deepcopy(spec)
    fixture=OvertakeFixture(world,[],spec,events)
    assert spec==original
    assert fixture.spec['activate_m']==30
    assert fixture.spec['end_route_s_m']==120
    assert fixture.speed==14
    assert fixture.spec['steps']==spec['steps']
    assert fixture.adjustments['source_event_position_preserved'] is False


def test_prerequisite_cannot_be_silently_dropped(monkeypatch):
    world,spec,events=prepare(monkeypatch)
    spec['requires_task_success']=['prior']
    with pytest.raises(ConfigError,match='prerequisite'):
        OvertakeFixture(world,[],spec,events)


def test_hold_requires_continuous_condition_and_cleanup_attempts_all(monkeypatch):
    world,spec,events=prepare(monkeypatch)
    fixture=OvertakeFixture(world,[],spec,events)
    assert not fixture._held(True,0,1)
    assert not fixture._held(False,.9,1)
    assert not fixture._held(True,1,1)
    assert fixture._held(True,2,1)
    calls=[]
    fixture.actors=[NS(is_alive=True,destroy=lambda:False),
                    NS(is_alive=True,destroy=lambda:calls.append('second'))]
    with pytest.raises(RuntimeError,match='destroy'):
        fixture.destroy()
    assert calls==['second']
