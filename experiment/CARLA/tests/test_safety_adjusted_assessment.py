from copy import deepcopy
import pytest
from benchmark.safety_adjusted_assessment import SafetyAdjustedTaskOracle
from benchmark.task_oracle import TaskOracle
from experiment.CARLA.tests.test_task_oracle import spec,observation


def row(t,**ego):
    r=observation(t,**ego);frame=f"carla_{r['frame']}"
    r['execution_evidence']=dict(previous_pid_execution=dict(observation_frame_id=frame,
        action='keep_lane',effective_target_speed_kmh=10.,speed_target_status='CONSTRAINED',
        speed_constraint_codes=['junction_speed_cap'],pid={}),
        control_decision=dict(frame_id=frame,action='keep_lane'))
    return r


def test_adjusted_assessment_does_not_rewrite_strict_result():
    p=spec(dict(kind='speed',target_kmh=30,tolerance_kmh=2,hold_s=2,keep_lane=True,
        max_lateral_error_m=.35,max_heading_error_deg=5),dict(kind='progress',distance_m=30))
    strict=TaskOracle(p);adapted=SafetyAdjustedTaskOracle(p)
    for i in range(9):
        r=row(i*.5,speed_kmh=10.,route_s_m=i*10.)
        a=adapted.update(deepcopy(r));s=strict.update(deepcopy(r))
    assert a['status']=='SUCCESS' and s['status']=='RUNNING'
    assert a['evaluation_mode']=='SAFETY_ADJUSTED_ENGINEERING'and a['official_acceptance']is False
    assert a['speed_adjustments']


def test_standstill_is_not_adjusted_speed_completion():
    p=spec(dict(kind='speed',target_kmh=30,tolerance_kmh=2,hold_s=2))
    oracle=SafetyAdjustedTaskOracle(p)
    for i in range(7):r=oracle.update(row(i*.5,speed_kmh=0.))
    assert r['completed_steps']==0


@pytest.mark.parametrize('reason,expected',[('traffic_signal_stopline_constraint',True),('learned_visual_risk_high',False)])
def test_only_supported_safety_waits_pause_engineering_deadline(reason,expected):
    r=row(1.,speed_kmh=0.)
    r['execution_evidence']['control_decision'].update(action='stop',reason=reason)
    r['execution_evidence']['traffic_control_contract']=dict(observation=dict(applicable=True,
        state='RED',stop_line_source='static_hdmap_stop_line',stop_distance_m=2.,confidence=.99))
    assert SafetyAdjustedTaskOracle.justified_wait(r)==expected


def test_wait_credit_is_bounded_and_keeps_original_timestamps():
    p=spec(dict(kind='speed',target_kmh=30,tolerance_kmh=2,hold_s=2));p['timeout_s']=.2
    o=SafetyAdjustedTaskOracle(p,max_safety_wait_s=.5)
    for i in range(10):
        r=row(i*.1,speed_kmh=0.)
        r['execution_evidence']['control_decision'].update(action='stop',reason='traffic_signal_stopline_constraint')
        r['execution_evidence']['traffic_control_contract']=dict(observation=dict(applicable=True,state='RED',
            stop_line_source='static_hdmap_stop_line',stop_distance_m=2.,confidence=.99))
        result=o.update(r)
    assert result['status']=='TIMEOUT' and result['justified_safety_wait_s']==.5


def test_collision_still_fails_even_when_engineering_speed_would_pass():
    o=SafetyAdjustedTaskOracle(spec(dict(kind='speed',target_kmh=30,tolerance_kmh=2,hold_s=2)))
    o.update(row(0,speed_kmh=10.));r=row(.1,speed_kmh=10.);r['safety']['collisions']=1
    assert o.update(r)['status']=='FAILURE'


@pytest.mark.parametrize('valid',[True,False])
def test_relative_acceleration_can_be_bounded_by_real_geometry_not_model_labels(valid):
    p=spec(dict(kind='speed_change',direction='INCREASE',delta_kmh=20,hold_s=2))
    strict=TaskOracle(p);adapted=SafetyAdjustedTaskOracle(p)
    for i in range(6):
        r=row(i*.5,speed_kmh=6.)
        e=r['execution_evidence']['previous_pid_execution'];e['effective_target_speed_kmh']=7.
        if not valid:e['speed_constraint_codes']=['learned_visual_risk_medium']
        a=adapted.update(deepcopy(r));s=strict.update(deepcopy(r))
    assert (a['status']=='SUCCESS')==valid
    assert s['status']=='RUNNING'


@pytest.mark.parametrize('filename',['benchmark/summary.json','vla_control_decisions.jsonl'])
def test_assessment_tool_cannot_replace_input_records(tmp_path,filename):
    from experiment.CARLA.tools.assess_safety_adjusted_run import validate_output
    with pytest.raises(ValueError):validate_output(tmp_path,tmp_path/filename)


def test_assessment_tool_does_not_replace_unrelated_existing_documents(tmp_path):
    from experiment.CARLA.tools.assess_safety_adjusted_run import validate_output
    import json
    output=tmp_path/'report.json';output.write_text(json.dumps({'keep':'original'}))
    with pytest.raises(ValueError):validate_output(tmp_path/'run',output)


def test_assessment_sidecar_is_a_separate_permitted_output(tmp_path):
    from experiment.CARLA.tools.assess_safety_adjusted_run import validate_output
    validate_output(tmp_path,tmp_path/'engineering_assessment.json')


@pytest.mark.parametrize('supported',[False,True])
def test_a_physical_stop_flag_alone_does_not_pause_the_deadline(supported):
    r=row(1.,speed_kmh=0.);r['actors']={}
    r['execution_evidence']['control_decision']['action']='stop'
    r['execution_evidence']['risk']=dict(frame_id=f"carla_{r['frame']}",
        risk_evidence=dict(is_fresh=True),behavior_policy=dict(physical_stop=True,
            required_clearance_m=10.,lidar=dict(observed_at_s=1.,nearest_obstacle_m=2. if supported else None)))
    assert SafetyAdjustedTaskOracle.justified_wait(r)==supported
