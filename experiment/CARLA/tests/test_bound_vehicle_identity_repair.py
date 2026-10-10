from collections import deque
import numpy as np
from control.passing_conditions import PassingMemory, short_curve_score


def track(x=10.):
    descriptor = np.array([1., 0.])
    return dict(category='vehicle', position=np.array([x, 0.]), velocity=np.array([4., 0.]),
        velocity_valid=True, last_seen_s=1., history=deque([(i*.1, np.array([x-4.+i*.4, 0.]))
            for i in range(11)], maxlen=80), appearances=deque([descriptor], maxlen=8),
        image_aliases={('front', 'image-a'): 1.}, motion_history_established=True)


def measurement(x=10.4, color=None):
    return dict(category='vehicle', position=np.array([x, 0., 1.]), view='front',
        local_id='image-a', box_cropped=False, surface=np.array([[x+i*.01, 0., 1.] for i in range(4)]),
        appearance=np.array([1., 0.]) if color is None else np.asarray(color))


def test_bound_image_alias_restores_split_world_track_with_current_body():
    memory = PassingMemory()
    memory.tracks = {'target': track(), 'split': track(10.4)}
    memory.bound[('command', 'pass')] = dict(target_id='target')
    memory.aliases[('front', 'image-a')] = 'split'
    memory.update([measurement()], frame=100, timestamp_s=1.1, ego_xy=(0, 0), heading=(1, 0))
    assert memory.aliases[('front', 'image-a')] == 'target'
    assert memory.tracks['target']['frame'] == 100
    assert memory.tracks['target']['identity_repair_source'] == 'measured_bound_image_alias_continuity'
    assert not memory.facts


def test_other_clothing_or_large_position_jump_cannot_restore_target():
    for current in [measurement(color=[0., 1.]), measurement(x=30.)]:
        memory = PassingMemory()
        memory.tracks = {'target': track(), 'split': track(current['position'][0]-.4)}
        memory.bound[('command', 'pass')] = dict(target_id='target')
        memory.aliases[('front', 'image-a')] = 'split'
        memory.update([current], frame=100, timestamp_s=1.1, ego_xy=(0, 0), heading=(1, 0))
        assert memory.tracks['target'].get('frame') != 100
        assert not memory.facts


def test_two_bound_identities_are_not_merged_by_alias_prior():
    memory = PassingMemory()
    memory.tracks = {'first': track(), 'second': track(10.4)}
    memory.bound = {('command', 'first'): dict(target_id='first'),
                    ('command', 'second'): dict(target_id='second')}
    memory.aliases[('front', 'image-a')] = 'second'
    memory.update([measurement()], frame=100, timestamp_s=1.1, ego_xy=(0, 0), heading=(1, 0))
    assert memory.aliases[('front', 'image-a')] == 'second'
    assert memory.tracks['first'].get('frame') != 100


def test_short_curved_road_handover_uses_static_geometry_not_ego_tangent():
    road = np.array([[0., 0., 0.], [10., 0., 0.], [20., 0., 0.], [20., 10., 0.], [20., 20., 0.]])
    old = track(19.)
    old['velocity'] = np.array([5., 0.])
    current = measurement(); current['position'] = np.array([20., 4., 1.])
    current['surface'] = np.array([[20., 4., 1.]]*4)
    assert PassingMemory.surface_association(old, current, 2., np.array([1., 0.]), np.array([0., 1.])) is None
    assert short_curve_score(old, current, 2., road) is not None
    current['position'][1] += 3.01
    assert short_curve_score(old, current, 2., road) is None
