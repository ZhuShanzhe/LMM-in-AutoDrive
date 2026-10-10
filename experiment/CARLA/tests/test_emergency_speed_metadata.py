"""Emergency telemetry is fresh without changing the actuator command."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS


def test_emergency_branch_replaces_previous_speed_telemetry():
    tree=ast.parse((Path(__file__).resolve().parents[1]/'control/pid_controller.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='EgoPIDController')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='run_step')
    scope={'normalize_intent':lambda intent,_:intent,
        'carla':NS(VehicleControl=lambda **kw:NS(**kw))}
    exec(compile(ast.Module(body=[method],type_ignores=[]),'<actual-pid-method>','exec'),scope)
    controller=NS(default_speed_kmh=30.,_emergency_latched=False,_emergency_clear_frames=0,
        _lateral_control=lambda intent,dt:.1,_reset_longitudinal_state=lambda:None,
        _last_effective_target_speed_kmh=30.,_last_speed_constraint_codes=['old_cap'])
    control,_=scope['run_step'](controller,dict(action='emergency_brake',emergency=True,
        target_speed_kmh=0.,allow_positive_acceleration=False),.05)
    assert control.throttle==0. and control.brake==1. and control.steer==.1
    assert controller._last_effective_target_speed_kmh==0.
    assert controller._last_controller_requested_speed_kmh==0.
    assert controller._last_speed_constraint_codes==['emergency_brake']
