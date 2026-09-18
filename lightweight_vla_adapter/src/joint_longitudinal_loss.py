"""Measured multi-parameter longitudinal supervision, including normal 1s holds."""

import torch
from torch.nn import functional as F

DT = .1
HOLD_STEPS = 10
OBJECTIVE_VERSION = 'measured_joint_longitudinal/2.0'
LOSS_WEIGHTS = dict(speed=1., acceleration=.5, first_acceleration=1., command=.3,
                    overlap_speed=.5, overlap_acceleration=.1, risk_preservation=.2,
                    jerk=.1, displacement=.2, held_speed=.5, held_acceleration=.3)


def normal_hold_projection(acceleration, initial_speed, desired_speed):
    """Reference at a normal operation boundary, not a vehicle dynamics simulator."""
    held = acceleration[:, :HOLD_STEPS].mean(1).clamp(-8., 3.)
    times = torch.arange(1, HOLD_STEPS + 1, device=held.device, dtype=held.dtype) * DT
    ceiling = torch.maximum(initial_speed.clamp_min(0), desired_speed.clamp_min(0))
    speed = torch.minimum((initial_speed[:, None].clamp_min(0) + held[:, None] * times).clamp_min(0),
                          ceiling[:, None])
    return speed, held


def trajectory_displacement(speed, initial_speed):
    previous = torch.cat((initial_speed[:, None], speed[:, :-1]), dim=1)
    return torch.cumsum(.5 * (previous + speed) * DT, dim=1)


def _masked_huber(predicted, target, eligible):
    per_sample = F.smooth_l1_loss(predicted, target, reduction='none')
    if per_sample.ndim > 1:
        per_sample = per_sample.flatten(1).mean(1)
    return (per_sample * eligible).sum() / eligible.sum().clamp_min(1)


def joint_longitudinal_loss(out, nxt, batch, prior):
    speed = out['speed_sequence_mps']
    a = out['acceleration_sequence_mps2']
    target = batch['speed_target'][:, 0].to(speed)
    target_a = batch['accel_target'][:, 0].to(speed)
    initial = batch['initial_speed_mps'][:, 0].to(speed)
    desired = batch['desired_speed_mps'].to(speed)
    held_speed, held_a = normal_hold_projection(a, initial, desired)
    times = torch.arange(1, HOLD_STEPS + 1, device=speed.device, dtype=speed.dtype) * DT
    # Least-squares constant acceleration approximates the measured first second.
    target_held_a = ((target[:, :HOLD_STEPS] - initial[:, None]) * times).sum(1) / times.square().sum()
    normal = (target_a[:, :HOLD_STEPS].amin(1) > -3.).to(speed)
    terms = dict(
        speed=F.smooth_l1_loss(speed / (3 / 3.6), target / (3 / 3.6)),
        acceleration=F.smooth_l1_loss(a, target_a),
        first_acceleration=F.smooth_l1_loss(a[:, 0], target_a[:, 0]),
        command=F.smooth_l1_loss(out['commanded_acceleration_mps2'], target_a),
        overlap_speed=F.smooth_l1_loss(speed[:, 1:], nxt['speed_sequence_mps'][:, :-1]),
        overlap_acceleration=F.smooth_l1_loss(a[:, 1:], nxt['acceleration_sequence_mps2'][:, :-1]),
        risk_preservation=F.kl_div(out['risk_logits'].log_softmax(-1), prior['risk_logits'].softmax(-1), reduction='batchmean'),
        # Match observed jerk; a zero-jerk penalty would suppress genuine braking.
        jerk=F.smooth_l1_loss(torch.diff(a, dim=1) / (DT * 3.), torch.diff(target_a, dim=1) / (DT * 3.)),
        displacement=F.smooth_l1_loss(trajectory_displacement(speed, initial), trajectory_displacement(target, initial)),
        held_speed=_masked_huber(held_speed / (3 / 3.6), target[:, :HOLD_STEPS] / (3 / 3.6), normal),
        held_acceleration=_masked_huber(held_a, target_held_a, normal),
    )
    return sum(LOSS_WEIGHTS[key] * value for key, value in terms.items()), terms


def joint_longitudinal_metrics(out, batch):
    speed = out['speed_sequence_mps']
    a = out['acceleration_sequence_mps2']
    target = batch['speed_target'][:, 0].to(speed)
    target_a = batch['accel_target'][:, 0].to(speed)
    initial = batch['initial_speed_mps'][:, 0].to(speed)
    held_speed, held_a = normal_hold_projection(a, initial, batch['desired_speed_mps'].to(speed))
    return dict(jerk_mae_mps3=(torch.diff(a, dim=1) - torch.diff(target_a, dim=1)).abs().mean(1) / DT,
                displacement_endpoint_mae_m=(trajectory_displacement(speed, initial)[:, -1] - trajectory_displacement(target, initial)[:, -1]).abs(),
                held_reference_speed_mae_kmh=(held_speed - target[:, :HOLD_STEPS]).abs().mean(1) * 3.6,
                held_mean_acceleration_mae_mps2=(held_a - target_a[:, :HOLD_STEPS].mean(1)).abs())
