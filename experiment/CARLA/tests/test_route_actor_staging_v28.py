from types import SimpleNamespace as NS
import sys
import pytest
from scenarios.complex.town05_scene2 import DeterministicSceneEvents


def make_manager():
    event=dict(id='compound',kind='route_slow_vehicle',actor_role='lead',
               activate_progress_m=880.,anchor_progress_m=980.,
               complete_progress_m=1350.,staging_lead_m=80.,
               target_speed_kmh=18.,retain_after_completion=True,
               ground_truth={'actor_roles':['lead']})
    manager=DeterministicSceneEvents(None,None,None,[],[],[event],42)
    manager.states['bus_stop_passengers']='STAGED'
    calls=[]
    def spawn(event):
        calls.append('spawn')
        manager.route_slow_vehicles[event['id']]=NS(is_alive=True)
        manager.spawn_diagnostics[event['actor_role']]={}
    manager._spawn_route_slow_vehicle=spawn
    manager._activate_route_slow_vehicle=lambda event:calls.append('release')
    return manager,event,calls


def test_no_early_physical_actor_and_exactly_one_release():
    manager,event,calls=make_manager()
    manager.update(0.)
    manager.update(799.9)
    assert calls==[]
    manager.update(800.)
    manager.update(850.)
    assert calls==['spawn']
    assert manager.spawn_diagnostics['lead']['staged_at_progress_m']==800.
    manager.update(880.)
    manager.update(900.)
    assert calls==['spawn','release']
    manager.update(1350.)
    assert manager.states['compound']=='RESOLVED'
    assert manager.route_slow_vehicles['compound'].is_alive


@pytest.mark.parametrize('start',[0.,799.9,800.,870.,900.])
def test_cold_start_stages_before_first_activation(start):
    manager,event,calls=make_manager()
    assert manager.skip_events_before(start)==[]
    assert calls==(['spawn'] if start>=800. else [])
    manager.update(max(start,880.))
    assert calls==['spawn','release']


def test_late_segment_skips_without_creating_actor():
    manager,event,calls=make_manager()
    assert manager.skip_events_before(1600.)==['compound']
    manager.update(1600.)
    assert calls==[] and not manager.route_slow_vehicles


@pytest.mark.parametrize('lead',[-1.,float('nan'),float('inf')])
def test_invalid_staging_horizon_rejected(lead):
    manager,event,calls=make_manager()
    event['staging_lead_m']=lead
    with pytest.raises(ValueError):
        manager._stage_route_slow_vehicle(event,800.)
    assert calls==[]


def test_release_sets_public_route_without_relocating_actor(monkeypatch):
    monkeypatch.setitem(sys.modules,'carla',NS(VehicleControl=lambda **kw:NS(**kw)))
    manager,event,calls=make_manager()
    manager.route=[(NS(transform=NS(location=i)),None) for i in range(1400)]
    manager.distances=list(range(1400))
    actor=NS(is_alive=True,apply_control=lambda control:calls.append('control'),
             set_autopilot=lambda *args:calls.append('autopilot'))
    manager.route_slow_vehicles['compound']=actor
    manager.spawn_diagnostics['lead']={}
    manager.traffic_manager=NS(get_port=lambda:8000,
        set_desired_speed=lambda *args:None,auto_lane_change=lambda *args:None,
        update_vehicle_lights=lambda *args:None,
        set_path=lambda actor,path:calls.append(('path',path)))
    DeterministicSceneEvents._activate_route_slow_vehicle(manager,event)
    path=next(value for kind,value in (c for c in calls if isinstance(c,tuple)) if kind=='path')
    assert path[0]==980 and path[-1]==1399
    assert manager.spawn_diagnostics['lead']['route_path_end_progress_m']==1399
    assert manager.spawn_diagnostics['lead']['release_progress_m']==880.
