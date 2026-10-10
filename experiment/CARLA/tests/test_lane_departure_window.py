from types import SimpleNamespace
import pytest
from control.lane_departure_window import lane_departure_window,relative_lane_rebind_allowed


class Waypoint:
    lane_id=-1
    is_junction=False

    def __init__(self, hazard=None):self.hazard=hazard

    def next(self,distance):
        if self.hazard=='branch' and distance>=8.:return [self,self]
        if self.hazard=='end' and distance>=8.:return []
        return [SimpleNamespace(is_junction=self.hazard=='junction' and distance>=5.,
            lane_id=1 if self.hazard=='opposing' and distance>=5. else -1)]


def test_nonjunction_unique_lanes_allow_departure():
    assert lane_departure_window(Waypoint(),Waypoint())['allowed']


@pytest.mark.parametrize('hazard',['branch','end','junction','opposing'])
@pytest.mark.parametrize('side',['current','target'])
def test_departure_requires_both_lane_horizons(hazard,side):
    a,b=Waypoint(),Waypoint()
    if side=='current':a=Waypoint(hazard)
    else:b=Waypoint(hazard)
    assert not lane_departure_window(a,b)['allowed']


@pytest.mark.parametrize('horizon',[0.,float('nan'),float('inf'),True])
def test_invalid_horizon_fails_closed(horizon):
    assert not lane_departure_window(Waypoint(),Waypoint(),horizon_m=horizon)['allowed']


def test_missing_target_fails_closed():
    assert not lane_departure_window(Waypoint(),None)['allowed']


@pytest.mark.parametrize('explicit',[None,'return_to','target_ref','lane_id','target','selectors'])
def test_only_plain_unstarted_relative_direction_can_rebind(explicit):
    step=dict(action='CHANGE_LANE',parameters=dict(direction='LEFT'))
    if explicit in {'target','selectors'}:step[explicit]='anchored reference'
    elif explicit is not None:step['parameters'][explicit]='anchored reference'
    document=dict(intent=dict(steps=[step]));state=dict(active_step_index=0)
    assert relative_lane_rebind_allowed(document,state)==(explicit is None)


def test_unresolved_or_completed_step_cannot_rebind():
    assert not relative_lane_rebind_allowed(dict(intent=dict(steps=[])),dict(active_step_index=None))
