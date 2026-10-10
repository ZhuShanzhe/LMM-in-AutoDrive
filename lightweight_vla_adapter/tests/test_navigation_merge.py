from copy import deepcopy
from types import SimpleNamespace
import pytest
from lightweight_vla_adapter.src.navigation_merge import rear_departure_window, defer_navigation_merge
from lightweight_vla_adapter.tests.test_runtime_gate_contract import deferred


class Waypoint:
    lane_id = -1
    is_junction = False

    def __init__(self, hazard=None):
        self.hazard = hazard

    def previous(self, distance):
        if self.hazard == 'branch' and distance >= 5.:
            return [self, self]
        if self.hazard == 'end' and distance >= 5.:
            return []
        return [SimpleNamespace(is_junction=self.hazard == 'junction' and distance >= 5.,
            lane_id=1 if self.hazard == 'opposing' and distance >= 5. else -1)]


def test_unique_rear_window_allows_observation():
    assert rear_departure_window(Waypoint(), Waypoint())['allowed']


@pytest.mark.parametrize('hazard', ['branch', 'end', 'junction', 'opposing'])
@pytest.mark.parametrize('side', ['current', 'target'])
def test_both_rear_corridors_must_be_unambiguous(hazard, side):
    current, target = Waypoint(), Waypoint()
    if side == 'current':
        current = Waypoint(hazard)
    else:
        target = Waypoint(hazard)
    assert not rear_departure_window(current, target)['allowed']


@pytest.mark.parametrize('horizon', [0., float('nan'), float('inf'), True])
def test_invalid_rear_horizon_fails_closed(horizon):
    assert not rear_departure_window(Waypoint(), Waypoint(), horizon_m=horizon)['allowed']


def context():
    decision, risk = deferred()
    evidence = decision.pop('deferred_current_lane_evidence')
    decision.pop('deferred_lane_change')
    decision.update(action='stop', target_speed_kmh=0., decision_status='BLOCKED',
        allow_positive_acceleration=False, longitudinal_control_mode='STOP',
        reason='requested_lane_wait_for_sensor_clearance')
    risk['target_lane_contract'].update(can_execute=False,
        departure_window=dict(allowed=False, rear_window=dict(allowed=False,
            reason='rear_departure_junction_horizon')))
    risk['lane_change']['left'].update(is_safe=False)
    return decision, risk, evidence


def test_defer_moves_only_in_verified_current_lane_without_clearing_target_gate():
    decision, risk, evidence = context()
    before = deepcopy(risk)
    result, reason = defer_navigation_merge(decision, risk, 'left',
        navigation_recovery=True, corridor_evidence=evidence)
    assert reason == 'navigation_merge_deferred_current_lane'
    assert result['action'] == 'keep_lane' and result['target_speed_kmh'] == 10.
    assert result['target_lane'] is None and result['target_location'] is None
    assert risk == before and risk['target_lane_contract']['can_execute'] is False


@pytest.mark.parametrize('reason', ['departure_branch_or_endpoint', 'departure_junction_horizon',
    'departure_direction_discontinuity'])
def test_forward_topology_wait_also_defers_without_blind_merge(reason):
    decision, risk, evidence = context()
    risk['target_lane_contract']['departure_window'] = dict(allowed=False, reason=reason)
    result, override = defer_navigation_merge(decision, risk, 'left',
        navigation_recovery=True, corridor_evidence=evidence)
    assert override and result['action'] == 'keep_lane' and result['target_lane'] is None


@pytest.mark.parametrize('bad', ['explicit', 'started', 'physical_stop', 'emergency',
    'path_blocked', 'wrong_frame', 'expired', 'missing_evidence', 'other_reason', 'rear_clear',
    'unknown_rear', 'bad_cap', 'nan_cap'])
def test_deferral_never_bypasses_safety_or_explicit_maneuver(bad):
    decision, risk, evidence = context()
    kwargs = dict(navigation_recovery=True, corridor_evidence=evidence)
    if bad == 'explicit': kwargs['navigation_recovery'] = False
    if bad == 'started': kwargs['maneuver_started'] = True
    if bad == 'physical_stop': risk['behavior_policy']['physical_stop'] = True
    if bad == 'emergency': decision['emergency'] = True
    if bad == 'path_blocked': risk['task_feasibility']['PATH_CLEAR']['can_execute'] = False
    if bad == 'wrong_frame': evidence['frame_id'] = 'old'
    if bad == 'expired': evidence['valid_until_s'] = 4.
    if bad == 'missing_evidence': kwargs['corridor_evidence'] = None
    if bad == 'other_reason': decision['reason'] = 'physical_stop'
    if bad == 'rear_clear': risk['target_lane_contract']['departure_window']['rear_window']['allowed'] = True
    if bad == 'unknown_rear': risk['target_lane_contract']['departure_window']['rear_window']['reason'] = 'rear_departure_geometry_unavailable'
    if bad == 'bad_cap': risk['behavior_policy']['max_speed_kmh'] = True
    if bad == 'nan_cap': risk['behavior_policy']['max_speed_kmh'] = float('nan')
    result, reason = defer_navigation_merge(decision, risk, 'left', **kwargs)
    assert reason is None and result == decision
