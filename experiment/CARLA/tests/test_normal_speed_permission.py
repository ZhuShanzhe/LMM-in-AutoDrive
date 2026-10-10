import ast
from pathlib import Path
from types import SimpleNamespace
import pytest


tree=ast.parse((Path(__file__).resolve().parents[1]/'control/pid_controller.py').read_text())
cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='EgoPIDController')
method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='_enforce_guarded_crawl_ceiling')
namespace={}
exec(compile(ast.Module(body=[method],type_ignores=[]),'<speed-bound-method>','exec'),namespace)
enforce=namespace['_enforce_guarded_crawl_ceiling']
run_step=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='run_step')
expressions={node.targets[0].id:compile(ast.Expression(node.value),'<speed-mode>','eval')
    for node in run_step.body if isinstance(node,ast.Assign)
    and isinstance(node.targets[0],ast.Name)
    and node.targets[0].id in {'risk_speed_cap','explicit_goal','guarded_crawl'}}


def test_normal_speed_permission_does_not_inherit_crawl_torque_limit():
    pid=SimpleNamespace(_speed_integral=1.)
    control=SimpleNamespace(throttle=.6,brake=0.)
    enforce(pid,control,30.,27.)
    assert control.throttle==.6


def test_crawl_torque_limit_allows_breakaway_but_is_bounded():
    control=SimpleNamespace(throttle=.6,brake=0.)
    enforce(SimpleNamespace(_speed_integral=1.),control,10.,8.)
    assert control.throttle==pytest.approx(.45)


def test_low_speed_ceiling_does_not_raise_a_small_pid_command():
    control=SimpleNamespace(throttle=.15,brake=0.)
    enforce(SimpleNamespace(_speed_integral=1.),control,7.,0.)
    assert control.throttle==.15


@pytest.mark.parametrize('target',[6.,16.,30.,60.])
def test_current_speed_ceiling_always_has_final_authority(target):
    pid=SimpleNamespace(_speed_integral=1.)
    control=SimpleNamespace(throttle=.6,brake=0.)
    enforce(pid,control,target,target+.2)
    assert control.throttle==0. and control.brake>=.1 and pid._speed_integral<=0.


@pytest.mark.parametrize('mode,allow,tracking',[
    ('ABSOLUTE_SPEED',True,True),('CAUTIOUS_SPEED_CAP',True,True),
    ('CAUTIOUS_SPEED_CAP',False,False),('GUARDED_CRAWL',True,False),
])
def test_explicit_goal_tracks_the_authorized_speed_not_the_unbounded_request(mode,allow,tracking):
    intent={'longitudinal_control_mode':mode,'speed_tracking_mode':'EXPLICIT_CRUISE',
        'allow_positive_acceleration':allow}
    variables={'intent':intent}
    variables['risk_speed_cap']=eval(expressions['risk_speed_cap'],variables)
    assert eval(expressions['explicit_goal'],variables) is tracking
    assert eval(expressions['guarded_crawl'],variables) is (mode in {'GUARDED_CRAWL','CAUTIOUS_SPEED_CAP'})
