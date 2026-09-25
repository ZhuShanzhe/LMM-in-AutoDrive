import pytest

from benchmark.catalog import ConfigError
from benchmark.task_oracle import TaskOracle


def definition():
    return dict(schema_version='task_oracle/1.0',task_id='parallel',activate_m=0,
                timeout_s=10,max_frame_gap_s=1,
                steps=[dict(kind='speed',target_kmh=30,tolerance_kmh=2,hold_s=1)],
                constraints=[dict(id='section_speed',kind='maintain_interval',from_route_s_m=0,
                                  until_route_s_m=30,max_sample_distance_m=12,max_speed_kmh=40)])


def row(t,speed=30):
    return dict(schema_version='task_truth/1.0',source='simulator_truth',frame=t,
                sim_time_s=t,scenario_valid=True,ego=dict(route_s_m=t*10,speed_kmh=speed),
                safety=dict(collisions=0,violations=0))


def test_task_waits_for_interval_after_steps_finished():
    oracle=TaskOracle(definition())
    oracle.update(row(0))
    result=oracle.update(row(1))
    assert result['completed_steps']==1 and result['status']=='RUNNING'
    oracle.update(row(2))
    result=oracle.update(row(3))
    assert result['status']=='SUCCESS'
    assert result['constraints']['section_speed']['status']=='SUCCESS'


def test_violation_after_action_success_still_fails_task():
    oracle=TaskOracle(definition())
    oracle.update(row(0))
    oracle.update(row(1))
    result=oracle.update(row(2,50))
    assert result['status']=='FAILURE'
    assert result['reason']=='constraint:section_speed:interval_speed_ceiling_exceeded'


def test_finished_constraint_cannot_finish_unfinished_action():
    spec=definition()
    spec['steps'][0]['target_kmh']=60
    oracle=TaskOracle(spec)
    for t in range(4):
        result=oracle.update(row(t))
    assert result['status']=='RUNNING'
    assert result['constraints']['section_speed']['status']=='SUCCESS'


def test_constraints_have_independent_fixture_and_state():
    spec=definition()
    spec['constraints'].append(dict(id='lane',kind='maintain_interval',from_route_s_m=0,
        until_route_s_m=30,max_sample_distance_m=12,keep_lane=True,
        max_lateral_error_m=.35,max_heading_error_deg=5))
    oracle=TaskOracle(spec)
    for t in range(4):
        observation=row(t)
        observation['ego'].update(lane_key='a',lateral_error_m=0,heading_error_deg=0)
        observation['fixture']={'constraints':{'lane':{'lane_corridor':[
            dict(start_m=0,end_m=30,lane_keys=['a'])]}}}
        result=oracle.update(observation)
    assert result['status']=='SUCCESS'
    assert len([e for e in result['evidence'] if e['event']=='CONSTRAINT_RESULT'])==2


def test_duplicate_constraint_id_rejected():
    spec=definition()
    spec['constraints']*=2
    with pytest.raises(ConfigError):
        TaskOracle(spec)
