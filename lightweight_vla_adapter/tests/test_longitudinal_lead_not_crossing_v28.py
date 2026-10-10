import pytest
from lightweight_vla_adapter.src.risk_behavior import potential_path_intrusion


def track(velocity,position=(10.,0.)):
    return dict(kind='UNCLASSIFIED_RADAR_SURFACE',status='TRACKED',age_s=.05,
        position_xy_m=list(position),velocity_xy_mps=list(velocity),
        observed_velocity_xy_mps=list(velocity),position_std_m=.18,
        motion_observation_span_s=.55,motion_sample_count=12,
        motion_position_residual_m=.035)


def check(entity,speed=0.):
    return potential_path_intrusion({'entities':[entity]},[(0.,0.),(40.,0.)],
        half_width_m=1.1,ego_front_m=1.85,speed_mps=speed)


@pytest.mark.parametrize('velocity',[(0.,0.),(.003,.05),(3.,0.)])
def test_distant_same_lane_lead_is_not_a_crossing_while_ego_waits(velocity):
    assert not check(track(velocity))


@pytest.mark.parametrize('velocity',[(0.,0.),(3.,0.)])
def test_close_physical_occupant_remains_protected(velocity):
    assert check(track(velocity,position=(5.,0.)))


@pytest.mark.parametrize('position,velocity', [((10.,0.),(0.,-.5)),((6.,4.),(0.,-3.))])
def test_confirmed_transverse_entrant_remains_protected(position,velocity):
    assert check(track(velocity,position=position),speed=1.)


def test_distant_fast_crossing_that_clears_before_ego_arrival_is_not_emergency():
    assert not check(track((0.,-3.),position=(10.,0.)),speed=1.)


def test_tentative_surface_cannot_invent_future_crossing_velocity():
    entity=track((0.,-3.));entity['status']='TENTATIVE'
    assert not check(entity)
