"""Offline longitudinal rollout supervision, NOT calibrated real-world risk.

State columns: ego speed, acceleration, front gap/speed/acceleration,
rear gap/speed/acceleration, friction estimate, desired speed (SI units).
Missing actors have gap=1000. Hypotheses are explicit modeling assumptions.
"""

import numpy as np

ACTION_NAMES = ('accelerate', 'keep_lane', 'decelerate', 'stop', 'emergency_brake')
ACCELERATIONS = np.array([2., 0., -2.5, -4.5, -8.], dtype=np.float32)


def rollout_targets(states):
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 10 or not np.isfinite(states).all():
        raise ValueError('Expected finite physical states [N,10]')
    n = len(states)
    shape = (n, 5, 3)
    ego_v = np.broadcast_to(states[:, 0, None, None], shape).copy()
    front_v = np.broadcast_to(states[:, 3, None, None], shape).copy()
    rear_v = np.broadcast_to(states[:, 6, None, None], shape).copy()
    ego_x = np.zeros(shape, dtype=np.float32)
    front_x = np.broadcast_to(states[:, 2, None, None], shape).copy()
    rear_x = np.broadcast_to(-states[:, 5, None, None], shape).copy()
    front_hit = np.zeros(shape, dtype=np.bool_)
    rear_hit = np.zeros(shape, dtype=np.bool_)
    severity = np.zeros(shape + (2,), dtype=np.float32)
    horizons = []
    friction = np.clip(states[:, 8, None, None] + np.array([-.1, 0, .1]), .2, 1.)
    delay = np.array([.6, .4, .2])
    for step in range(40):
        t = step * .1
        command = np.maximum(ACCELERATIONS[None, :, None], -9.81 * friction)
        acceleration = np.where(t >= delay, command, states[:, 1, None, None])
        # Do not accelerate through the requested speed; no actual controller outputs are labels.
        acceleration = np.where((ego_v >= states[:, 9, None, None]) & (acceleration > 0), 0, acceleration)
        lead_a = states[:, 4, None, None] * np.exp(-t / 2) + np.array([-1., 0., 1.])
        rear_a = states[:, 7, None, None] * np.exp(-t / 2) + np.array([1., 0., -1.])
        ego_next = np.maximum(0, ego_v + acceleration * .1)
        front_next = np.maximum(0, front_v + lead_a * .1)
        rear_next = np.maximum(0, rear_v + rear_a * .1)
        ego_x += (ego_v + ego_next) * .05
        front_x += (front_v + front_next) * .05
        rear_x += (rear_v + rear_next) * .05
        front_hit |= (front_x - ego_x <= 0) & (states[:, 2, None, None] < 200)
        rear_hit |= (ego_x - rear_x <= 0) & (states[:, 5, None, None] < 200)
        severity[..., 0] = np.maximum(severity[..., 0], np.maximum(ego_next - front_next, 0) / 20 * front_hit)
        severity[..., 1] = np.maximum(severity[..., 1], np.maximum(rear_next - ego_next, 0) / 20 * rear_hit)
        ego_v, front_v, rear_v = ego_next, front_next, rear_next
        if step in (9, 19, 39):
            horizons.append(np.stack((front_hit[:, 1].mean(-1), rear_hit[:, 1].mean(-1)), -1))
    collision = front_hit.astype(float) + rear_hit.astype(float)
    tracking = np.abs(ego_v - states[:, 9, None, None]) / 10
    cost = (100 * collision + 30 * severity.sum(-1) + tracking
            + np.abs(ACCELERATIONS[None, :, None]) * .025).mean(-1)
    selected = cost.argmin(-1)
    safe_front = front_hit.mean(-1) == 0
    required = np.full(n, 8., dtype=np.float32)
    for j in (4, 3, 2, 1):
        required = np.where(safe_front[:, j], max(0, -ACCELERATIONS[j]), required)
    horizon = np.stack(horizons, -1).astype(np.float32)
    danger = (horizon.max((1, 2)) >= 2 / 3) | (required >= 4.5)
    caution = (horizon.max((1, 2)) > 0) | (required >= 2.5)
    risk = np.where(danger, 2, np.where(caution, 1, 0))
    return dict(action=selected.astype(np.int64), risk=risk.astype(np.int64),
                horizon=horizon, severity=np.clip(severity[:, 1].mean(1), 0, 1).astype(np.float32),
                required_decel=required,
                speed=np.where(selected >= 3, 0, np.minimum(states[:, 9],
                    ego_v[np.arange(n), selected].mean(-1))).astype(np.float32),
                action_cost=cost.astype(np.float32))
