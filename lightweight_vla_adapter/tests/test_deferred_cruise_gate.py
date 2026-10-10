import pytest

from lightweight_vla_adapter.src.driving_plan_runtime import DrivingPlanRuntime
from lightweight_vla_adapter.src.lane_risk_contract import enforce_requested_lane
from lightweight_vla_adapter.src.safety_bridge import gate_vla_proposal
from lightweight_vla_adapter.tests.fixtures import proposal
from lightweight_vla_adapter.tests.test_runtime_gate_contract import setup


def context():
    runtime, canonical, risk = setup()
    canonical.update(decision_status='BLOCKED', action='stop', target_lane=None,
                     blocked_reason_codes=['passing_target_not_acquired'])
    evidence = dict(request_id=canonical['request_id'], frame_id=canonical['frame_id'],
        condition='CURRENT_LANE_MOTION', valid=True, satisfied=True,
        source='fresh_lidar_radar_driving_corridor', observed_at_s=5., valid_until_s=5.25,
        sensor_frame=20, required_clearance_m=2.5,
        semantic_scope='bounded_current_lane_motion_not_semantic_path_clear')
    risk['risk_evidence'].update(is_fresh=True, sensor_frame=20)
    risk['behavior_policy'].update(current_speed_required_clearance_m=2.5,
        lidar=dict(valid=True, clear_distance_m=35., sensor_frame=20, observed_at_s=5.))
    risk['physical_forward_radar'] = dict(sensor_frame=20, obstacle_candidate_count=0,
                                         nearest_distance_m=None)
    result = runtime.deferred_lane_navigation(canonical, risk, evidence, cruise_speed_kmh=30.)
    assert result['target_speed_kmh'] == 30.
    return result, risk


def test_faster_current_lane_approach_survives_both_interfaces_without_merge():
    canonical, risk = context()
    result = gate_vla_proposal(proposal(action='accelerate'), canonical, risk)
    result, reason = enforce_requested_lane(result, risk, 'left')
    assert reason == 'deferred_current_lane_only'
    assert result['action'] == 'keep_lane' and result['target_speed_kmh'] == 30.
    assert result['target_lane'] is None and result['target_location'] is None


@pytest.mark.parametrize('bad', ['short', 'radar', 'stale', 'cap', 'speed', 'lateral'])
def test_bridge_independently_rejects_unproven_faster_approach(bad):
    canonical, risk = context()
    if bad == 'short': risk['behavior_policy']['lidar']['clear_distance_m'] = 8.
    if bad == 'radar': risk['physical_forward_radar'].update(obstacle_candidate_count=1, nearest_distance_m=8.)
    if bad == 'stale': risk['physical_forward_radar']['sensor_frame'] = 19
    if bad == 'cap': risk['behavior_policy']['max_speed_kmh'] = 20.
    if bad == 'speed': canonical['target_speed_kmh'] = 31.
    if bad == 'lateral': canonical['action'] = 'lane_change_left'
    with pytest.raises(ValueError, match='invalid canonical'):
        gate_vla_proposal(proposal(action='accelerate'), canonical, risk)
