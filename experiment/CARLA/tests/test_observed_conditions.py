from types import SimpleNamespace as NS

import pytest

from control.observed_conditions import observed_condition_readiness, read_execution_feedback


@pytest.mark.parametrize('name',['execution_state','get_execution_state'])
def test_execution_feedback_reader_accepts_both_controller_contracts(name):
    controller=NS(**{name:lambda:{'pid':{'current_lane_id':-2}}})
    assert read_execution_feedback(controller)['pid']['current_lane_id']==-2


@pytest.mark.parametrize('fault',[None,'stale','missing_support','obstacle','high','wrong_sensor','identity'])
def test_wait_predicate_uses_fresh_physical_corridor_not_event_state(fault):
    doc=dict(request_id='p',intent={'steps':[dict(step_id='s',action='WAIT',parameters={
        'condition':'PEDESTRIAN_CLEAR' if fault=='identity' else 'PATH_CLEAR'})]})
    risk=dict(frame_id='f',risk_level='high' if fault=='high' else 'low',
        risk_evidence={'is_fresh':True,'sensor_frame':10},
        physical_forward_radar={'sensor_frame':11 if fault=='wrong_sensor' else 10,
            'obstacle_candidate_count':1 if fault=='obstacle' else 0,
            'nearest_distance_m':2. if fault=='obstacle' else None},
        behavior_policy={'disposition':'CLEAR','physical_stop':fault=='obstacle',
            'current_speed_required_clearance_m':5.,'lidar':{'valid':True,'sensor_frame':10,
                'observed_at_s':0. if fault=='stale' else 1.,
                'clear_distance_m':0. if fault=='missing_support' else 20.}})
    item=observed_condition_readiness(doc,risk,frame_id='f',timestamp_s=1.1)['s']
    assert item['satisfied']==(fault is None)
    assert item['source']=='fresh_lidar_radar_driving_corridor'
