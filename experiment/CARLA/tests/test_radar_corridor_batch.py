import math

import numpy as np
import pytest

from universal_vla_controller import filter_forward_radar_to_route_corridor


def scalar_members(bins, road, *, x, y, yaw, width):
    angle = math.radians(yaw)
    origin = (x + 2 * math.cos(angle), y + 2 * math.sin(angle))
    result = []
    for raw in bins:
        radius = float(raw['distance_m'])
        direction = angle + math.radians(raw['azimuth_deg'])
        point = (origin[0] + radius * math.cos(direction), origin[1] + radius * math.sin(direction))
        distances = []
        for first, last in zip(road, road[1:]):
            dx, dy = last[0] - first[0], last[1] - first[1]
            norm = dx * dx + dy * dy
            t = max(0., min(1., ((point[0] - first[0]) * dx + (point[1] - first[1]) * dy) / norm)) if norm > 1e-9 else 0.
            distances.append(math.hypot(point[0] - first[0] - t * dx, point[1] - first[1] - t * dy))
        distance = min(distances)
        if distance <= max(.5, width):
            result.append(dict(raw, route_lateral_distance_m=round(distance, 3)))
    return result


@pytest.mark.parametrize('seed', range(12))
def test_batched_filter_preserves_scalar_geometry(seed):
    rng = np.random.default_rng(seed)
    road = np.cumsum(rng.uniform(-3, 5, (90, 2)), axis=0).tolist()
    road[8] = road[7][:]
    x, y, yaw = float(rng.normal()), float(rng.normal()), float(rng.uniform(-180, 180))
    bins = [dict(distance_m=float(rng.uniform(.01, 100)), azimuth_deg=float(rng.uniform(-180, 180)),
        closing_speed_mps=float(rng.uniform(-5, 5)), relative_velocity_mps=float(rng.uniform(-5, 5))) for _ in range(700)]
    expected = scalar_members(bins, road, x=x, y=y, yaw=yaw, width=1.5)
    result = filter_forward_radar_to_route_corridor(dict(azimuth_obstacle_bins=bins),
        ego_x=x, ego_y=y, ego_yaw_deg=yaw, route_polyline=road, corridor_half_width_m=1.5)
    assert result['route_corridor_obstacle_bins'] == expected
    assert result['obstacle_candidate_count'] == len(expected)
    assert result['closing_candidate_count'] == sum(item['closing_speed_mps'] > .5 for item in expected)


@pytest.mark.parametrize('width', [.5, 1.0005, 1.5, 2.5005])
def test_exact_boundary_and_rounding_ties(width):
    bins = [dict(distance_m=math.hypot(10., lateral), azimuth_deg=math.degrees(math.atan2(lateral, 10.)))
        for lateral in [width - 1e-10, width, width + 1e-10, 1.0005]]
    road = [(0., 0.), (0., 0.), (100., 0.)]
    expected = scalar_members(bins, road, x=0., y=0., yaw=0., width=width)
    result = filter_forward_radar_to_route_corridor(dict(azimuth_obstacle_bins=bins),
        ego_x=0., ego_y=0., ego_yaw_deg=0., route_polyline=road, corridor_half_width_m=width)
    assert result['route_corridor_obstacle_bins'] == expected


def test_empty_and_missing_bins_keep_fail_closed_contract():
    original = dict(nearest_distance_m=3., obstacle_candidate_count=1)
    assert filter_forward_radar_to_route_corridor(original, ego_x=0., ego_y=0., ego_yaw_deg=0.,
        route_polyline=[(0., 0.), (10., 0.)], corridor_half_width_m=1.) == original
    result = filter_forward_radar_to_route_corridor(dict(azimuth_obstacle_bins=[]),
        ego_x=0., ego_y=0., ego_yaw_deg=0., route_polyline=[(0., 0.), (10., 0.)], corridor_half_width_m=1.)
    assert result['route_corridor_obstacle_bins'] == []
    assert result['nearest_distance_m'] is None
