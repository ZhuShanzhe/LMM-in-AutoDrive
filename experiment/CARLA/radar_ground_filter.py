"""Conservative local surface rejection from synchronized physical sensors."""

from copy import deepcopy
import math
import numpy as np
from scipy.spatial import cKDTree


def filter_ground(observation, lidar_world, sensor_matrix):
    result = deepcopy(observation)
    points = result.pop('_ground_candidates', [])
    result.pop('_sensor_matrix', None)
    result.pop('_filtered', None)
    diagnostics = dict(status='unavailable', rejected=0, examined=0, retained_reasons={})
    result['ground_filter'] = diagnostics
    if lidar_world is None or sensor_matrix is None:
        return result
    cloud = np.asarray(lidar_world, dtype=float)
    matrix = np.asarray(sensor_matrix, dtype=float)
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    if len(cloud) < 8 or matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return result
    tree = cKDTree(cloud[:, :2])
    diagnostics['status'] = 'same_frame_local_plane'

    def retain(reason):
        counts=diagnostics['retained_reasons'];counts[reason]=counts.get(reason,0)+1
        return False

    def is_ground(item):
        diagnostics['examined'] += 1
        az, alt = np.deg2rad([item['azimuth_deg'], item['altitude_deg']])
        ray = np.array([math.cos(alt)*math.cos(az), math.cos(alt)*math.sin(az), math.sin(alt)])
        point = matrix[:3, :3] @ (ray*item['distance_m']) + matrix[:3, 3]
        near = cloud[tree.query_ball_point(point[:2], 2.5)]
        if len(near) < 8:
            return retain('insufficient_support')
        xy = near[:, :2]-point[:2]
        # Require spatial support on both sides, not extrapolation from a wall.
        if not ((xy.min(axis=0) < -.25).all() and (xy.max(axis=0) > .25).all()):
            return retain('extrapolation')
        a = np.column_stack([xy, np.ones(len(near))])
        fit, _, rank, singular = np.linalg.lstsq(a, near[:, 2], rcond=None)
        residual = a@fit-near[:, 2]
        if (rank < 3 or singular[-1] <= 0 or singular[0]/singular[-1] > 100
                or np.linalg.norm(fit[:2]) > .25 or np.sqrt(np.mean(residual**2)) > .015
                or np.max(np.abs(residual)) > .04):
            return retain('nonplanar_or_ill_conditioned')
        # A roof/top surface must not become a ground hypothesis. Mixed
        # ground/object neighborhoods fail the residual test above.
        origin_surface = fit[2] + fit[:2] @ (matrix[:2, 3]-point[:2])
        # Test the plane at the sensor origin, not the elevated distant
        # return: a valid uphill road can rise above the sensor's own height.
        if not .5 <= matrix[2, 3]-origin_surface <= 2.5:
            return retain('surface_above_ground_band')
        if abs(point[2]-fit[2]) > .02:return retain('above_surface')
        return True

    kept = []
    for item in points:
        if is_ground(item):
            diagnostics['rejected'] += 1
        else:
            kept.append(item)
    result['tracking_points'] = [{k: p[k] for k in ('distance_m', 'relative_velocity_mps', 'azimuth_deg', 'altitude_deg')} for p in kept]
    obstacles = [p for p in kept if abs(p['azimuth_deg']) <= 8]
    bins = {}
    for p in obstacles:
        key = round(p['azimuth_deg'])
        if key not in bins or p['distance_m'] < bins[key]['distance_m']:
            bins[key] = p
    result['azimuth_obstacle_bins'] = [bins[k] for k in sorted(bins)]
    closing = [p for p in obstacles if p['closing_speed_mps'] > .5]
    nearest = min(obstacles, key=lambda p:p['distance_m'], default={})
    approaching = min(closing, key=lambda p:p['distance_m'], default={})
    result.update(obstacle_candidate_count=len(obstacles), closing_candidate_count=len(closing),
        nearest_distance_m=nearest.get('distance_m'), nearest_relative_velocity_mps=nearest.get('relative_velocity_mps'),
        nearest_azimuth_deg=nearest.get('azimuth_deg'), nearest_relative_height_m=nearest.get('relative_height_m'),
        nearest_altitude_deg=nearest.get('altitude_deg'), nearest_closing_distance_m=approaching.get('distance_m'),
        nearest_closing_velocity_mps=approaching.get('closing_speed_mps'))
    return result
