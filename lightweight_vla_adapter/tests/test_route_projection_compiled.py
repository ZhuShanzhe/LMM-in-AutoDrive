import numpy as np
import pytest
from lightweight_vla_adapter.src import route_projection_compiled as kernel
from lightweight_vla_adapter.src import risk_behavior
from control import passing_conditions


@pytest.fixture(scope='module')
def prepared():
    assert kernel.prepare_route_projection(), 'CPU compiler is needed for this integration test'


@pytest.mark.parametrize('seed',range(20))
def test_full_functions_are_identical_with_compiled_backend(prepared,seed):
    rng=np.random.default_rng(seed)
    road=np.cumsum(np.r_[np.zeros((1,3)),np.c_[rng.uniform(.2,2.,(42,2)),rng.uniform(-.02,.02,42)]],axis=0)
    points=rng.uniform(-5.,40.,(4000,3))
    function=kernel._kernel
    kernel._kernel=None
    try:
        before=risk_behavior.lidar_corridor_evidence(points,road,half_width_m=1.6,
            ego_front_m=2.4,sensor_frame=12,observed_at_s=1.2)
        surface=passing_conditions.object_surface_points(points,road)
    finally:
        kernel._kernel=function
    after=risk_behavior.lidar_corridor_evidence(points,road,half_width_m=1.6,
        ego_front_m=2.4,sensor_frame=12,observed_at_s=1.2)
    assert before==after
    np.testing.assert_array_equal(surface,passing_conditions.object_surface_points(points,road))


def test_empty_and_equal_distance_first_segment_tie(prepared):
    road=np.array([[0.,0.,0.],[10.,0.,1.],[0.,0.,2.]])
    result=kernel.project_points_to_route(np.empty((0,3)),road)
    assert all(v.shape==(0,) for v in result)
    lateral,progress,height=kernel.project_points_to_route([[5.,1.,2.]],road)
    np.testing.assert_array_equal(lateral,[1.])
    np.testing.assert_array_equal(progress,[5.])
    np.testing.assert_array_equal(height,[1.5])


@pytest.mark.parametrize('points,road',[
    ([[0.,0.,float('nan')]],[[0.,0.,0.],[1.,0.,0.]]),
    ([[0.,0.,0.]],[[0.,0.,0.],[0.,0.,0.]]),
    ([[0.,0.]],[[0.,0.,0.],[1.,0.,0.]]),
    ([[0.,0.,0.]],[[0.,0.,0.]]),
])
def test_invalid_geometry_is_rejected(prepared,points,road):
    with pytest.raises(ValueError):kernel.project_points_to_route(points,road)


def test_frame_function_never_compiles_implicitly(prepared,monkeypatch):
    monkeypatch.setattr(kernel,'_kernel',None)
    with pytest.raises(RuntimeError):kernel.project_points_to_route([[0.,0.,0.]],[[0.,0.,0.],[1.,0.,0.]])
