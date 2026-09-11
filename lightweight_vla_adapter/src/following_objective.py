"""Relative-motion supervision using measured lead futures as labels, never inputs."""

import torch
from torch.nn import functional as F

from .joint_longitudinal_loss import joint_longitudinal_loss, trajectory_displacement

OBJECTIVE_VERSION = 'relative_following_behavior/1.0'


def following_terms(out, batch):
    speed = out['speed_sequence_mps']
    initial = batch['initial_speed_mps'][:, 0].to(speed)
    lead = batch['lead_future_mps'].to(speed)
    lead_initial = batch['lead_initial_mps'].to(speed)
    initial_gap = batch['initial_gap_m'].to(speed)
    usable = batch['following_valid'].to(speed)
    gap = initial_gap[:, None] + trajectory_displacement(lead, lead_initial) - trajectory_displacement(speed, initial)
    desired_gap = 6. + 1.4 * speed
    tolerance = torch.maximum(torch.full_like(desired_gap, 3.), desired_gap * .25)
    error = gap - desired_gap
    # A feasible catch-up envelope, not a demand to remove a large gap in 3 seconds.
    initial_error = initial_gap - (6. + 1.4 * initial)
    closure = (initial - lead_initial).clamp_min(0)
    times = torch.arange(1, 31, device=speed.device, dtype=speed.dtype) * .1
    permitted_excess = (initial_error.clamp_min(0)[:, None] -
        torch.minimum(closure[:, None] * times + .5 * times.square(), initial_error.clamp_min(0)[:, None])).clamp_min(0)
    too_far = (error - torch.maximum(tolerance, permitted_excess)).relu()
    too_close = (-error - tolerance).relu()
    desired_relative = (.35 * error).clamp(-3., 3.)
    relative = speed - lead
    near_weight = (1. - error.abs() / (desired_gap + 3.)).clamp(.2, 1.)
    # Penalize closing momentum which cannot be removed with the assumed 2 m/s^2 deceleration.
    closing = relative.relu()
    reserve = gap - 2. - .5 * closing - closing.square() / 4.
    risk = (-reserve).relu()
    def masked(value):
        per_sample = value.mean(1)
        return (per_sample * usable).sum() / usable.sum().clamp_min(1)
    terms = dict(gap_convergence=masked(F.smooth_l1_loss(too_far / 10., torch.zeros_like(too_far), reduction='none')),
        gap_near=masked(F.smooth_l1_loss(too_close / 3., torch.zeros_like(too_close), reduction='none')),
        relative_speed=masked(near_weight * F.smooth_l1_loss(relative, desired_relative, reduction='none')),
        approach_risk=masked(F.smooth_l1_loss(risk, torch.zeros_like(risk), reduction='none')))
    return terms, dict(predicted_gap_m=gap, predicted_closing_mps=relative, predicted_reserve_m=reserve)


def following_loss(out, nxt, batch, prior):
    base, components = joint_longitudinal_loss(out, nxt, batch, prior)
    # The old 1-second mean-acceleration hold is not the new endpoint executor.
    base = base - .5 * components['held_speed'] - .3 * components['held_acceleration']
    terms, _ = following_terms(out, batch)
    weights = dict(gap_convergence=1., gap_near=2., relative_speed=.5, approach_risk=2.)
    gate = F.binary_cross_entropy_with_logits(out['extend_memory_logits'], batch['extend_target'].to(base))
    reconstruction = F.smooth_l1_loss(out['state_reconstruction'], batch['memory'][:,0,-1].to(base))
    ego = batch['initial_speed_mps'][:,0].to(base)
    lead = batch['lead_initial_mps'].to(base)
    gap = batch['initial_gap_m'].to(base)
    closing = (ego-lead).relu()
    state_target = torch.stack(((gap-6.-1.4*ego)/30.,(ego-lead)/5.,
        (2.+.5*closing+closing.square()/4.-gap).relu()/10.),-1)
    state_error = F.smooth_l1_loss(out['relative_motion_state'],state_target,reduction='none').mean(1)
    eligible = batch['following_valid'].to(base)
    motion_state = (state_error*eligible).sum()/eligible.sum().clamp_min(1)
    total = base + sum(weights[k] * v for k, v in terms.items()) + .15 * gate + .1 * reconstruction + .2 * motion_state
    return total, dict(**components, **terms, extend_memory=gate, state_reconstruction=reconstruction,
                      relative_motion_state=motion_state)
