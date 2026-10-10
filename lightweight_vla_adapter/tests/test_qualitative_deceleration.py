from copy import deepcopy
import pytest
from lightweight_vla_adapter.src.driving_plan_runtime import DrivingPlanRuntime, qualitative_deceleration_reached
from lightweight_vla_adapter.tests.fixtures import integration_documents


@pytest.mark.parametrize('extra',[{}, {'target_speed_kmh':15.}, {'target_speed_mps':15./3.6},
                                  {'speed_delta_kmh':5.}, {'speed_delta_mps':5./3.6}])
def test_qualitative_only_not_numeric_targets(extra):
    step=dict(action='ADJUST_SPEED',parameters=dict(change='DECREASE',**extra))
    assert qualitative_deceleration_reached(step,dict(speed_reference_kmh=20.,resolved_target_speed_kmh=15.),8.,1.) == (not extra)


@pytest.mark.parametrize('bad',[None,True,float('nan'),float('inf'),-1.,'8'])
def test_invalid_measurement_never_completes(bad):
    assert not qualitative_deceleration_reached(dict(action='ADJUST_SPEED',parameters=dict(change='DECREASE')),
        dict(speed_reference_kmh=20.,resolved_target_speed_kmh=15.),bad,1.)


@pytest.mark.parametrize('fault',[None,'stale','wrong_step','no_decrease','gap','explicit_target','numeric_delta'])
def test_runtime_requires_fresh_continuous_qualitative_slowdown(fault):
    document,world,_,risk=integration_documents(parser_action='ADJUST_SPEED')
    step=document['intent']['steps'][0]
    step.update(parameters=dict(change='DECREASE'),completion=dict(type='TARGET_SPEED_REACHED'))
    if fault=='explicit_target':step['parameters']['target_speed_kmh']=15.
    if fault=='numeric_delta':step['parameters']['speed_delta_kmh']=5.
    second=deepcopy(step)
    second.update(step_id='step_2',action='STOP',parameters={},depends_on=['step_1'],
                  trigger=dict(type='AFTER_STEP',step_id='step_1'),completion=dict(type='VEHICLE_STOPPED'))
    document['intent']['steps'].append(second)
    runtime=DrivingPlanRuntime(speed_completion_hold_s=.5)
    for index in range(10):
        frame=f'v106-{index}'
        world['frame_id']=risk['frame_id']=frame
        speed=20. if index==0 or fault=='no_decrease' else 8.
        world['ego']['speed_mps']=speed/3.6
        evidence=dict(source_step_id='other' if fault=='wrong_step' else 'step_1',
                      observation_frame_id='old' if fault=='stale' else frame)
        runtime.prepare(document,frame_id=frame,timestamp_s=index*(.3 if fault=='gap' else .1),
                        speed_mps=speed/3.6,execution_state=evidence)
        runtime.advance(world,risk)
    assert (runtime.state['step_states'][0]['status']=='COMPLETED') == (fault is None)
