"""Scene-fixture isolation; never provides targets or permission to ego policy."""
import math


def _nearest_segment(position, route):
    best = None
    for left, right in zip(route, route[1:]):
        dx, dy = right[0] - left[0], right[1] - left[1]
        length2 = dx * dx + dy * dy
        if length2 < 1e-6:
            continue
        raw = ((position.x - left[0]) * dx + (position.y - left[1]) * dy) / length2
        fraction = max(0., min(1., raw))
        x, y = left[0] + fraction * dx, left[1] + fraction * dy
        distance2 = (position.x - x) ** 2 + (position.y - y) ** 2
        if best is None or distance2 < best[0]:
            length = math.sqrt(length2)
            best = (distance2, raw, dx / length, dy / length,
                left[2] + fraction * (right[2] - left[2]),
                left[3] + fraction * (right[3] - left[3]))
    return best


def prune_unrelated_task_corridor(route, vehicles, *, owned_ids, excluded_ids, protected_roles):
    """Remove owned ambient bodies in the configured same-direction task corridor.

    Route rows contain x, y, z and lane width. Endpoint, height and heading checks
    preserve unrelated roads, opposing traffic and traffic outside the task window.
    """
    if len(route) < 2:
        return []
    if any(len(row) != 4 or not all(math.isfinite(float(v)) for v in row)
            or row[3] <= 0 for row in route):
        raise ValueError('Invalid task corridor geometry')
    owned, excluded, protected = set(owned_ids), set(excluded_ids), set(protected_roles)
    removed = []
    for actor in vehicles:
        role = actor.attributes.get('role_name', '')
        if (not actor.is_alive or actor.id not in owned or actor.id in excluded
                or role in protected or role in {'hero', 'ego'}):
            continue
        pose = actor.get_transform()
        nearest = _nearest_segment(pose.location, route)
        if nearest is None:
            continue
        distance2, fraction, tx, ty, height, width = nearest
        yaw = math.radians(pose.rotation.yaw)
        alignment = math.cos(yaw) * tx + math.sin(yaw) * ty
        if not all(math.isfinite(v) for v in (distance2, fraction, alignment, pose.location.z)):
            raise ValueError('Invalid background actor geometry')
        if fraction < 0. or fraction > 1. or abs(pose.location.z - height) > 3. or alignment < .5:
            continue
        # One adjacent same-direction lane is part of the passing approach.
        box = actor.bounding_box
        normal_support = (abs(-ty * math.cos(yaw) + tx * math.sin(yaw)) * box.extent.x
            + abs(ty * math.sin(yaw) + tx * math.cos(yaw)) * box.extent.y)
        if math.sqrt(distance2) > 1.5 * width + normal_support:
            continue
        record = dict(actor_id=int(actor.id), role=role,
            position_xyz_m=[pose.location.x, pose.location.y, pose.location.z],
            yaw_deg=pose.rotation.yaw, half_length_m=box.extent.x, half_width_m=box.extent.y,
            reason='unrelated_background_in_configured_task_approach', policy_input=False,
            source='authorized_scene_simplification')
        if not actor.destroy():
            raise RuntimeError('Failed to remove conflicting task-corridor background actor')
        removed.append(record)
    return removed
