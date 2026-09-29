import sys
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from scenarios.complex.town05_scene2 import DeterministicSceneEvents


@pytest.mark.parametrize('activation,lead,expected',[(1500,200,1300),(900,100,800),(30,200,0),(1234,150,1084)])
def test_staging_uses_event_contract_not_a_route_identity(activation,lead,expected):
    assert DeterministicSceneEvents._staging_progress(dict(
        activate_progress_m=activation,anchor_progress_m=activation+75,
        staging_lead_m=lead))==expected


@pytest.mark.parametrize('lead',[-1,float('nan'),float('inf')])
def test_invalid_staging_lead_rejected(lead):
    with pytest.raises(ValueError):
        DeterministicSceneEvents._staging_progress(dict(
            activate_progress_m=500,anchor_progress_m=600,staging_lead_m=lead))


def test_future_actor_not_staged_on_earlier_pass_and_released_normally(monkeypatch):
    manager=object.__new__(DeterministicSceneEvents)
    event=dict(id='cycle_event',kind='cyclist',activate_progress_m=500,
               anchor_progress_m=600,staging_lead_m=100,target_speed_kmh=14)
    manager.events=[event]
    manager.states={'cycle_event':'STAGED','bus_stop_passengers':'STAGED'}
    manager.selected_variants={'cycle_event':'v'}
    manager.scripted_walkers={}
    manager.traffic_manager=Mock()
    manager._set_desired_speed=Mock()
    manager.cyclist=None
    manager._cyclist_placed=False
    def prestage(config, progress):
        if manager.cyclist is not None:
            manager._cyclist_placed=True
    manager._prestage_cyclist=Mock(side_effect=prestage)
    actor=Mock(is_alive=True)
    def stage(config,progress):
        if manager.cyclist is None:
            manager.cyclist=actor
            manager.cyclist_transform=object()
    manager._stage_cyclist=Mock(side_effect=stage)
    monkeypatch.setitem(sys.modules,'carla',SimpleNamespace(VehicleControl=lambda:object()))
    manager.update(100)
    manager._stage_cyclist.assert_not_called()
    manager.update(400)
    assert manager.cyclist is actor
    actor.set_autopilot.assert_not_called()
    manager.update(500)
    assert manager.states['cycle_event']=='ACTIVE'
    actor.set_autopilot.assert_called_once_with(True,manager.traffic_manager.get_port())
    manager._set_desired_speed.assert_called_once_with(actor,14.,75.)
