import torch

from lightweight_vla_adapter.src.event_observation import prepare_event_radar, prepare_event_ego
from lightweight_vla_adapter.src.event_memory import EventMemoryHead


def radar(*bins):
    return dict(sensor_frame=10, azimuth_obstacle_bins=list(bins))


def point(distance, speed, azimuth):
    return dict(distance_m=distance, relative_velocity_mps=speed, azimuth_deg=azimuth)


def test_off_corridor_return_does_not_replace_front_target():
    result = prepare_event_radar(radar(point(18, -4, 6.6), point(30, -2, 1)))
    assert result['nearest_distance_m'] == 30
    assert result['nearest_relative_velocity_mps'] == -2


def test_impossible_speed_is_unknown_not_empty_road():
    result = prepare_event_radar(radar(point(56, 2317, 0)))
    assert result['sensor_frame'] == -1
    assert result['nearest_distance_m'] is None


def test_fresh_empty_and_off_corridor_are_valid_empty():
    assert prepare_event_radar(radar())['sensor_frame'] == 10
    assert prepare_event_radar(radar(point(18, -4, 7)))['sensor_frame'] == 10


def test_closing_range_and_speed_belong_to_same_return():
    result = prepare_event_radar(radar(point(12, 2, 0), point(24, -3, 0)))
    assert result['nearest_distance_m'] == 12
    assert result['nearest_closing_distance_m'] == 24
    assert result['nearest_closing_velocity_mps'] == 3


def test_shared_ego_uses_causal_acceleration_and_no_actuator_leakage():
    ego = torch.tensor([[10., 9000, 0, .2, .5, .6, 15, 0]])
    memory = torch.zeros(1, 80, 52)
    memory[0, -1, 41] = -.25
    value = prepare_event_ego(ego, memory)
    assert value[0, 1] == 3
    assert value[0, 3:6].sum() == 0
    assert ego[0, 1] == 9000


def test_hard_routing_has_identical_train_eval_forward():
    torch.manual_seed(1)
    head = EventMemoryHead()
    memory, valid = torch.randn(2, 80, 52), torch.ones(2, 80, dtype=torch.bool)
    trained = head.train()(memory, valid)
    evaluated = head.eval()(memory, valid)
    torch.testing.assert_close(trained['action_logits'], evaluated['action_logits'])
    trained['action_logits'].square().mean().backward()
    assert head.duration_head.weight.grad.abs().sum() > 0


# Label-generation tests remain with the private training worktree.
def test_planner_corridor_is_not_refiltered_as_heading_corridor():
    from lightweight_vla_adapter.src.event_observation import prepare_event_radar
    point=dict(distance_m=20.,relative_velocity_mps=-1.,azimuth_deg=12.)
    packet=dict(sensor_frame=4,azimuth_obstacle_bins=[point],route_corridor_filter_applied=True,
        route_corridor_obstacle_bins=[point],route_corridor_half_width_m=1.3)
    assert prepare_event_radar(packet)['nearest_distance_m'] is None
    preserved=prepare_event_radar(packet,preserve_route_corridor=True)
    assert preserved['nearest_distance_m']==20.
    assert preserved['event_corridor_mode']=='planner_route'
