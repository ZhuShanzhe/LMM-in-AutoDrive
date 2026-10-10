import numpy as np
from control.pedestrian_multiview import merge_cross_view_observations, measured_pedestrian_point
from control.pedestrian_identity import PedestrianIdentityAssociation
from control.pedestrian_conditions import PedestrianClearanceMemory


def person(name, view, x, y=0., appearance=None):
    return dict(track_id=f'{view}:{name}', view=view, position=[x, y, 0.],
                age_frames=5, appearance=np.array([1., 0.]) if appearance is None else appearance)


def test_unique_overlapping_views_share_identity():
    values = merge_cross_view_observations([person('a', 'front', 10), person('b', 'left', 10.1)])
    assert len(values) == 1 and set(values[0]['aliases']) == {'front:a', 'left:b'}
    association = PedestrianIdentityAssociation()
    identity = association.update(values, 1.)[0]['track_id']
    assert association.update([person('b', 'left', 10.2)], 1.1)[0]['track_id'] == identity


def test_three_and_four_unique_views_share_one_identity():
    for views in [('front', 'left', 'right'), ('front', 'left', 'right', 'rear')]:
        observations = [person(str(i), view, 10.+i*.1) for i, view in enumerate(views)]
        values = merge_cross_view_observations(observations)
        assert len(values) == 1 and len(values[0]['aliases']) == len(views)


def test_connected_but_inconsistent_group_does_not_transfer_identity():
    observations = [person('a', 'front', 10.), person('b', 'left', 10.5), person('c', 'rear', 11.)]
    assert len(merge_cross_view_observations(observations)) == 3


def test_same_view_neighbors_or_ambiguous_cross_views_not_merged():
    assert len(merge_cross_view_observations([person('a', 'front', 10), person('b', 'front', 10.1)])) == 2
    assert len(merge_cross_view_observations([person('a', 'front', 10),
        person('b', 'front', 10.2), person('c', 'left', 10.1)])) == 3


def test_different_clothing_cannot_fuse():
    assert len(merge_cross_view_observations([person('a', 'front', 10),
        person('b', 'left', 10.1, appearance=np.array([0., 1.]))])) == 2


def test_bounded_recovery_requires_measured_appearance_and_motion():
    association = PedestrianIdentityAssociation()
    for t in np.arange(1., 1.81, .1):
        identity = association.update([person('a', 'front', 10., t-1.)], float(t))[0]['track_id']
    recovered = association.update([person('b', 'left', 10., 3.)], 4.)[0]
    assert recovered['track_id'] == identity
    memory = PedestrianClearanceMemory()
    road = [[0., 0., 0.], [50., 0., 0.]]
    memory.update('event', [dict(track_id=identity, position=[10., 0., 0.])], road,
        half_width=1., timestamp_s=1., sensor_frame=1)
    result = memory.update('event', [recovered], road, half_width=1., timestamp_s=4., sensor_frame=2)
    assert not result['satisfied']
    for i, t in enumerate([4.1, 4.2, 4.3, 4.4, 4.6], 3):
        result = memory.update('event', [recovered], road, half_width=1., timestamp_s=t, sensor_frame=i)
    assert result['satisfied']


def test_recovery_rejects_reachability_and_wrong_clothing():
    for x, descriptor in [(30., np.array([1., 0.])), (10., np.array([0., 1.]))]:
        association = PedestrianIdentityAssociation()
        for t in np.arange(1., 1.81, .1):
            identity = association.update([person('a', 'front', 10.)], float(t))[0]['track_id']
        assert association.update([person('b', 'left', x, 0., descriptor)], 4.)[0]['track_id'] != identity


def test_measured_body_can_observe_sidewalk_beyond_front_road_support():
    pose = np.eye(4)
    pose[2, 3] = -1.5
    cloud = np.array([[10., -10., z] for z in [.7, .9, 1.1, 1.3, 1.5]])
    # Side-facing camera at y=-10, looking forward, with actual body returns.
    pose[1, 3] = 10.
    point = measured_pedestrian_point(cloud, pose, (100, 100), 90.,
        [40., 40., 60., 70.], [[0., 0., 0.], [50., 0., 0.]])
    assert point is not None and point[:2] == [10., -10.]
    assert measured_pedestrian_point(cloud[:3], pose, (100, 100), 90.,
        [40., 40., 60., 70.], [[0., 0., 0.], [50., 0., 0.]]) is None


def test_floor_returns_or_vehicle_sized_cluster_cannot_prove_exit():
    pose = np.eye(4); pose[2, 3] = -1.5
    road = [[0., 0., 0.], [50., 0., 0.]]
    floor = np.array([[10.+i*.02, 0., 0.] for i in range(10)])
    assert measured_pedestrian_point(floor, pose, (100, 100), 90., [40., 40., 60., 70.], road) is None
