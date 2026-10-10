import numpy as np
import pytest
import radar_ground_filter as module


@pytest.mark.parametrize('seed',range(50))
def test_valid_bounded_plane_is_never_pre_rejected(seed):
    rng=np.random.default_rng(seed)
    xy=rng.uniform(-2.5,2.5,(300,2))
    angle=rng.uniform(-np.pi,np.pi);slope=rng.uniform(0.,.25)
    z=xy@np.array([np.cos(angle),np.sin(angle)])*slope+rng.uniform(-.04,.04,300)
    assert not module._impossible_surface_plane(xy,z)


def test_vertical_mixture_cannot_satisfy_ground_plane_thresholds():
    assert module._impossible_surface_plane(np.array([[0.,0.],[.1,.1]]),np.array([0.,2.]))


def test_threshold_boundary_still_uses_original_solver():
    xy=np.array([[0.,0.],[2.,0.]])
    assert not module._impossible_surface_plane(xy,np.array([0.,.58+1e-9]))


@pytest.mark.parametrize('seed',range(30))
def test_complete_filter_output_matches_original_solver(monkeypatch,seed):
    rng=np.random.default_rng(seed)
    xy=rng.uniform([-5.,-8.],[30.,8.],(4000,2))
    height=.05*xy[:,0]+rng.normal(0.,.004,len(xy))
    height[rng.random(len(xy))<.15]+=rng.uniform(.5,2.5)
    cloud=np.column_stack([xy,height]);matrix=np.eye(4);matrix[2,3]=1.6
    candidates=[dict(distance_m=float(rng.uniform(3.,28.)),azimuth_deg=float(rng.uniform(-8.,8.)),
        altitude_deg=float(rng.uniform(-4.,4.)),relative_velocity_mps=-2.,
        closing_speed_mps=2.,relative_height_m=0.) for _ in range(40)]
    observation=dict(_ground_candidates=candidates,_sensor_matrix=matrix.tolist())
    actual=module.filter_ground(observation,cloud,matrix)
    with monkeypatch.context() as patch:
        patch.setattr(module,'_impossible_surface_plane',lambda *args:False)
        expected=module.filter_ground(observation,cloud,matrix)
    assert actual==expected
