import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from scenarios.complex.town05_scene2 import extend_event_corridor


@pytest.fixture(autouse=True)
def road_options(monkeypatch):
    monkeypatch.setitem(sys.modules, 'agents.navigation.local_planner',
        NS(RoadOption=NS(LANEFOLLOW='LANEFOLLOW')))


def waypoint(x, *, road=1, yaw=0.):
    return NS(road_id=road,lane_id=-1,s=x,
        transform=NS(location=NS(x=x,y=0.,z=0.),rotation=NS(yaw=yaw)))


def test_corridor_prefers_connected_straight_exit_and_keeps_prefix():
    first,straight,turn=waypoint(0),waypoint(2),waypoint(2,road=7)
    first.next=lambda d:[turn,straight]
    straight.next=lambda d:[waypoint(2+d)]
    turn.next=lambda d:[waypoint(2+d,road=7,yaw=90)]
    route,distances=extend_event_corridor([(first,None)],2.,2.)
    assert route[0]==(first,None)
    assert route[1][0] is straight
    assert distances==[0.,2.]


def test_corridor_never_teleports_across_a_disconnected_road_end():
    first=waypoint(0)
    first.next=lambda d:[]
    with pytest.raises(RuntimeError,match='disconnected lane end'):
        extend_event_corridor([(first,None)],10.,2.)


def test_simplified_route_preserves_required_length_events_and_mixed_traffic():
    path=Path(__file__).resolve().parents[1]/'configs/scene_2_town05_runtime.json'
    config=json.loads(path.read_text())
    assert config['route']['strategy']=='event_corridor_v2'
    assert config['route']['target_length_m']>=8000
    assert config['route']['corridor_start_m']>=max(
        event.get('resolve_progress_m',event['anchor_progress_m']) for event in config['special_events'])
    assert len(config['special_events'])==5
    assert config['traffic']['vehicles']>=55
    assert config['traffic']['buses']==0
    assert config['traffic']['ambient_walkers']>=1
    assert any(event['id']=='bus_stop_passengers' for event in config['special_events'])
    source=(path.parents[1]/'scenarios/complex/town05_scene2.py').read_text()
    assert 'self.bus = ' in source
    assert 'vehicle.mitsubishi.fusorosa' in source
    commands=config['commands']
    assert 'TURN:RIGHT' in commands[0]['steps']
    assert 'CHANGE_LANE:LEFT_WHEN_SAFE' in commands[2]['steps']
    assert all(not step.startswith(('TURN:','CHANGE_LANE:'))
        for command in commands[8:] for step in command['steps'])
    assert commands[-1]['steps'][-1]=='PROCEED:FINISH'
