from copy import deepcopy
import pytest
from lightweight_vla_adapter.src.maneuver_evidence import completed_lane_origin


def fixture():
    document=dict(request_id='request',intent=dict(steps=[dict(step_id='merge',action='CHANGE_LANE',
        parameters=dict(direction='LEFT'))]))
    facts=[dict(request_id='request',step_id='merge',condition='LANE_CHANGE_COMPLETED',
        source='held_physical_execution_feedback')]
    execution=dict(observation_frame_id='frame',pid=dict(lane_change_completed=True,
        current_lane_ref=dict(road_id=34,section_id=0,lane_id=3),target_lane_id=3,
        lane_change_command_id='request:merge',lateral_error_m=.03,heading_error_deg=.2))
    return document,facts,execution


def test_road_spanning_completion_recovers_origin_without_guessing_a_lane_key():
    document,facts,execution=fixture()
    before=deepcopy(execution)
    result=completed_lane_origin(document,facts,execution,'frame')
    assert result['origin_side']=='RIGHT'
    assert execution==before


@pytest.mark.parametrize('key,value',[('lane_change_completed',False),('emergency_latched',True),
    ('target_lane_id',4),('lane_change_command_id','different:merge'),('lateral_error_m',.36),
    ('heading_error_deg',5.1),('heading_error_deg',float('nan')),
    ('current_lane_ref',dict(lane_id=3))])
def test_unconfirmed_or_off_goal_execution_has_no_passing_permission(key,value):
    document,facts,execution=fixture();execution['pid'][key]=value
    assert completed_lane_origin(document,facts,execution,'frame') is None


@pytest.mark.parametrize('key,value',[('request_id','other'),('source','model_prediction'),
    ('condition','COMMAND_ISSUED'),('step_id','other')])
def test_fact_identity_and_physical_source_are_required(key,value):
    document,facts,execution=fixture();facts[0][key]=value
    assert completed_lane_origin(document,facts,execution,'frame') is None


def test_stale_execution_and_original_lane_return_are_not_new_passing_context():
    document,facts,execution=fixture()
    assert completed_lane_origin(document,facts,execution,'new_frame') is None
    document['intent']['steps'][0]['parameters']['return_to']='ORIGINAL_LANE'
    assert completed_lane_origin(document,facts,execution,'frame') is None
