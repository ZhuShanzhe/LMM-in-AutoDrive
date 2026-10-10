import numpy as np
import pytest
from radar_ground_filter import prepare_ground_surface, filter_ground, _candidate_position


def observation(direction=1):
    return {'_ground_candidates': [dict(distance_m=float(d), azimuth_deg=float(a),
        altitude_deg=-2., relative_height_m=-.3, relative_velocity_mps=0.,
        closing_speed_mps=0.) for d in (8, 10, 15) for a in (-4, 0, 4)]}


def sensor(direction):
    matrix=np.eye(4);matrix[2,3]=1.
    if direction<0:
        matrix[0,0]=matrix[1,1]=-1.
    return matrix


@pytest.mark.parametrize('seed',range(5))
def test_bounded_tree_preserves_both_exact_radius_neighborhoods(seed):
    rng=np.random.default_rng(seed)
    cloud=rng.uniform((-100,-100,-.01),(100,100,.01),size=(100000,3))
    regions=[(observation(),sensor(d)) for d in (1,-1)]
    baseline=prepare_ground_surface(cloud)
    candidate=prepare_ground_surface(cloud,query_regions=regions)
    assert len(candidate[0])<len(baseline[0])
    for obs,matrix in regions:
        assert filter_ground(obs,cloud,matrix,surface=baseline)==filter_ground(obs,cloud,matrix,surface=candidate)


def test_invalid_query_geometry_keeps_full_surface_not_partial_crop():
    cloud=np.zeros((20,3))
    matrix=sensor(1);matrix[0,0]=np.nan
    assert len(prepare_ground_surface(cloud,query_regions=[(observation(),matrix)])[0])==20


@pytest.mark.parametrize('yaw', [0., 37., 90., 179.])
def test_support_boundary_points_are_not_lost_by_grid_lookup(yaw):
    matrix=sensor(1)
    angle=np.deg2rad(yaw)
    matrix[:2,:2]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
    matrix[:2,3]=[-231.12,74.38]
    obs=observation()
    queries=np.array([_candidate_position(item,matrix)[:2] for item in obs['_ground_candidates']])
    offsets=np.array([[2.5,0.],[-2.5,0.],[0.,2.5],[0.,-2.5],[0.,0.]])
    xy=(queries[:,None,:]+offsets[None,:,:]).reshape(-1,2)
    cloud=np.column_stack([xy,np.zeros(len(xy))])
    cloud=np.concatenate([cloud,[[100.,100.,0.],[200.,200.,0.]]])
    baseline=prepare_ground_surface(cloud)
    candidate=prepare_ground_surface(cloud,query_regions=[(obs,matrix)])
    for query in queries:
        left=baseline[0][baseline[1].query_ball_point(query,2.5)]
        right=candidate[0][candidate[1].query_ball_point(query,2.5)]
        assert sorted(map(tuple,left))==sorted(map(tuple,right))
