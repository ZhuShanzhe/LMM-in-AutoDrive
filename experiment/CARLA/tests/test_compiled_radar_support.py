"""Compiled geometry must preserve the reference obstacle decisions."""
import math

import numpy as np
import pytest
import radar_ground_filter as candidate
import radar_support_precheck as support


def test_kernel_not_initialized_uses_reference(monkeypatch):
    monkeypatch.setattr(support, '_kernel', None)
    assert support.screen_support(np.zeros((8, 3)), [[0., 0., 0.]], [list(range(8))], np.eye(4)) is None


def test_nonfinite_matrix_cannot_enter_kernel():
    if not support.prepare_support_precheck():
        pytest.skip('No compiler; runtime retains NumPy')
    matrix = np.eye(4)
    matrix[0, 0] = np.nan
    assert support.screen_support(np.zeros((8, 3)), [[0., 0., 0.]], [list(range(8))], matrix) is None


def test_empty_packet_has_empty_codes():
    if not support.prepare_support_precheck():
        pytest.skip('No compiler; runtime retains NumPy')
    assert support.screen_support(np.zeros((8, 3)), [], [], np.eye(4)).shape == (0,)


def test_seeded_plane_and_obstacle_parity():
    if not support.prepare_support_precheck():
        pytest.skip('No compiler; runtime retains NumPy')
    rng = np.random.default_rng(20261009)
    xx, yy = np.meshgrid(np.linspace(-1.8, 1.8, 13), np.linspace(-1.8, 1.8, 13))
    cases = []
    for slope in (0., .24999999, .25, .25000001, .4):
        for height in (.49999999, .5, .50000001, 1., 2.49999999, 2.5, 2.50000001, 3.):
            for above in (0., .01999999, .02, .02000001, .5):
                cases.append((slope, height, above, 0., 1.))
    for _ in range(2500):
        cases.append((rng.uniform(-.5, .5), rng.uniform(.1, 3.5), rng.uniform(-.1, .7),
            rng.choice([0., .003, .014, .015, .016, .04, .2]), rng.choice([.00001, .01, .1, 1.])))
    # This snapshot prevents a previous fallback test from silently bypassing compilation.
    assert support._kernel is not None
    for index, (slope, height, above, noise, scale_y) in enumerate(cases):
        x, y = xx.ravel()+10, yy.ravel()*scale_y
        z = (x-10)*slope+rng.normal(0, noise, len(x))
        cloud = np.column_stack([x, y, z])
        matrix = np.eye(4)
        matrix[2, 3] = height
        delta = above-height
        point = dict(distance_m=math.hypot(10., delta), azimuth_deg=0.,
            altitude_deg=math.degrees(math.atan2(delta, 10.)), relative_velocity_mps=0.,
            closing_speed_mps=0., relative_height_m=delta)
        observation = dict(_ground_candidates=[point], _sensor_matrix=matrix.tolist())
        surface = candidate.prepare_ground_surface(cloud)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(support, '_kernel', None)
            before = candidate.filter_ground(observation, cloud, matrix, surface=surface)
        after = candidate.filter_ground(observation, cloud, matrix, surface=surface)
        assert before == after, (index, cases[index], before, after)
