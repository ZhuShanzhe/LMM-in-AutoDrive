import pytest
from types import SimpleNamespace as NS

from benchmark import turn_fixture
from benchmark.catalog import ConfigError,load_catalog
from benchmark.task_oracle import load_profile


def test_sequence_advances_cursor_including_intermediate_nonjunction_steps(monkeypatch):
    calls=[]
    def turn(route,direction,start,end):
        calls.append((direction,start,end))
        return dict(junction_end_m=start+50)
    def straight(route,world_map,location,start,end,angle):
        calls.append(('STRAIGHT',start,end))
        return dict(junction_end_m=start+40)
    monkeypatch.setattr(turn_fixture,'bind_route_turn',turn)
    monkeypatch.setattr(turn_fixture,'bind_straight_junction',straight)
    steps=[dict(kind='turn',direction='RIGHT'),dict(kind='speed'),
           dict(kind='straight_junction',max_heading_change_deg=20),dict(kind='turn',direction='RIGHT')]
    result=turn_fixture.bind_junction_sequence([],None,None,steps,100,500)
    assert list(result)==['0','2','3']
    assert calls[1][1]>result['0']['junction_end_m']
    assert calls[2][1]>result['2']['junction_end_m']


def test_mismatched_next_junction_is_not_skipped(monkeypatch):
    def mismatch(*args):
        raise ConfigError('next junction mismatched')
    monkeypatch.setattr(turn_fixture,'bind_route_turn',mismatch)
    with pytest.raises(ConfigError,match='mismatched'):
        turn_fixture.bind_junction_sequence([],None,None,[dict(kind='turn',direction='RIGHT')],0,100)


def test_new_profiles_bind_to_source():
    catalog=load_catalog('scene_2')
    for identity in ('s2_t05_cmd_13','s2_t05_cmd_15'):
        profile=load_profile(catalog,catalog.select(identity)[0])
        assert profile is not None
        expected=3
        assert len([s for s in profile['steps'] if s['kind'] in {'turn','straight_junction'}])==expected


def test_straight_entry_projection_tie_requires_interior_junction():
    route=[dict(distance_m=d,x=d,y=0,z=0,yaw=0,road_id=1 if d<10 else 2,
                section_id=0,lane_id=-1,is_junction=10<=d<20)
           for d in (0,5,10,15,20,25)]
    def waypoint(location):
        return NS(is_junction=location.x==15,junction_id=7)
    world_map=NS(get_waypoint=waypoint)
    result=turn_fixture.bind_straight_junction(route,world_map,
        lambda **kw:NS(**kw),0,25,20)
    assert result['junction_id']==7
    world_map.get_waypoint=lambda location:NS(is_junction=False,junction_id=7)
    with pytest.raises(ConfigError,match='junction geometry'):
        turn_fixture.bind_straight_junction(route,world_map,lambda **kw:NS(**kw),0,25,20)
