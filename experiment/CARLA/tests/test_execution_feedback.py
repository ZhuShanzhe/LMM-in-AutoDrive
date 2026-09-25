import pytest
from benchmark.execution_feedback import LaneExecutionFeedback


INTENT=dict(command_id='left',action='lane_change_left')
STATE=dict(lane_change_command_id='left',target_lane_id=2,current_lane_id=2,
           in_junction=False,lateral_error_m=.1,heading_error_deg=1.)


def test_requires_two_seconds_measured_stability_and_emits_once():
    tracker=LaneExecutionFeedback()
    for i in range(40): assert tracker.update(INTENT,STATE,i*.05) is None
    assert tracker.update(INTENT,STATE,2.)['command_id']=='left'
    assert tracker.update(INTENT,STATE,2.05) is None


@pytest.mark.parametrize('change',[{'lane_change_command_id':'other'}, {'in_junction':True},
    {'lateral_error_m':.36},{'heading_error_deg':5.1},{'current_lane_id':3},
    {'lateral_error_m':float('nan')},{'lane_change_completed':True,'heading_error_deg':20}])
def test_controller_completion_flag_does_not_override_measurements(change):
    tracker=LaneExecutionFeedback()
    for i in range(100): assert tracker.update(INTENT,{**STATE,**change},i*.05) is None


def test_gap_resets_hold():
    tracker=LaneExecutionFeedback()
    for i in range(30): tracker.update(INTENT,STATE,i*.05)
    assert tracker.update(INTENT,STATE,2.) is None
    for i in range(1,40): assert tracker.update(INTENT,STATE,2.+i*.05) is None


def test_new_command_cannot_inherit_stability():
    tracker=LaneExecutionFeedback()
    for i in range(30): tracker.update(INTENT,STATE,i*.05)
    assert tracker.update({**INTENT,'command_id':'right'},STATE,1.5) is None
