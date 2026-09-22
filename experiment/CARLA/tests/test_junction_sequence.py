import pytest

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
        assert len([s for s in profile['steps'] if s['kind'] in {'turn','straight_junction'}])==3
