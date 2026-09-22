import sys
from types import SimpleNamespace as NS

import pytest

from scenarios.complex.town05_scene2 import DeterministicSceneEvents


def runtime(alive=True):
    obj=object.__new__(DeterministicSceneEvents)
    controls=[]
    obj.bus=NS(is_alive=alive,apply_control=controls.append,set_autopilot=lambda *args:None)
    obj.bus_transform=NS(location=NS(x=10,y=20,z=.5))
    obj.traffic_manager=NS(get_port=lambda:8000)
    obj.spawn_diagnostics={'scene2_bus_stop_bus':{}}
    return obj,controls


def test_bus_activation_never_requires_teleport_or_physics_toggle(monkeypatch):
    monkeypatch.setitem(sys.modules,'carla',NS(VehicleControl=lambda **kw:kw))
    obj,controls=runtime()
    obj._activate_bus()
    assert controls[-1]['hand_brake'] is True
    assert obj.spawn_diagnostics['scene2_bus_stop_bus']['activation_source']=='physical_parked_bus_no_relocation'


def test_missing_bus_is_not_silently_ignored():
    obj,_=runtime(False)
    with pytest.raises(RuntimeError,match='disappeared'):
        obj._activate_bus()
