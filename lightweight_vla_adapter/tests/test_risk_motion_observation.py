import pytest

from lightweight_vla_adapter.src.risk_motion_observation import encode_motion_observation, CausalMotionHistory


def radar(frame=10, **extra):
    return {'schema_version': 'physical_front_radar/1.0', 'sensor_frame': frame,
            'nearest_distance_m': 20., 'nearest_relative_velocity_mps': -10., **extra}


def observation(frame=10, time=1., front=None):
    return encode_motion_observation(front if front is not None else radar(frame), {}, frame=frame, timestamp_s=time)


def test_closing_sign_and_pairwise_ttc():
    row = observation(front=radar(nearest_closing_distance_m=30., nearest_closing_velocity_mps=5.))
    assert row['values'][3:6] == pytest.approx([-.25, .25, .1])
    assert row['values'][7:10] == pytest.approx([.375, .125, .3])


def test_receding_target_does_not_have_collision_ttc():
    row = observation(front=radar(nearest_relative_velocity_mps=10.))
    assert row['values'][4:6] == [0., 1.]


@pytest.mark.parametrize('frame', [-1, 7, 11])
def test_missing_stale_and_future_sensor_is_not_clear_road(frame):
    row = observation(front=radar(frame))
    assert row['values'][0] == 0.
    assert row['valid_mask'][0] and not any(row['valid_mask'][1:10])


def test_valid_empty_sensor_differs_from_unavailable_sensor():
    empty = observation(front=radar(nearest_distance_m=None))
    assert empty['values'][:2] == [1., 0.]
    assert empty['valid_mask'][1] and not empty['valid_mask'][2]


def test_truth_only_fields_do_not_become_motion_features():
    row = observation(front={'risk_physics': {'gap_m': 10., 'ttc_s': .1}, 'risk_level': 'high'})
    assert not any(row['values'])


def test_history_never_repeats_a_frame_to_fill_padding():
    history = CausalMotionHistory()
    history.push(observation(), episode_id='a')
    assert not history.push(observation(), episode_id='a')
    _, _, mask = history.tensors()
    assert mask.tolist() == [False, False, False, True]


def test_history_copies_input_values():
    history = CausalMotionHistory()
    row = observation()
    history.push(row, episode_id='a')
    row['values'][2] = .99
    assert history.tensors()[0][-1, 2] == pytest.approx(.25)


def test_history_rejects_wrong_feature_order():
    row = observation()
    row['feature_names'].reverse()
    with pytest.raises(ValueError, match='order'):
        CausalMotionHistory().push(row, episode_id='a')


def test_history_keeps_training_cadence_with_faster_camera_frames():
    history = CausalMotionHistory()
    assert history.push(observation(), episode_id='a')
    assert not history.push(observation(11, 1.05), episode_id='a')
    assert history.push(observation(12, 1.1), episode_id='a')
    assert history.tensors()[2].sum() == 2


@pytest.mark.parametrize('frame,time,episode', [(11, 2., 'a'), (9, .9, 'a'), (11, 1.05, 'b')])
def test_history_resets_at_gaps_rollbacks_and_episode_boundaries(frame, time, episode):
    history = CausalMotionHistory()
    history.push(observation(), episode_id='a')
    history.push(observation(frame, time), episode_id=episode)
    assert history.tensors()[2].sum() == 1
