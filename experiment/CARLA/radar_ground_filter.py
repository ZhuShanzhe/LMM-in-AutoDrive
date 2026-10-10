"""Conservative local surface rejection from synchronized physical sensors."""

from copy import deepcopy
import math
import numpy as np
from scipy.spatial import cKDTree


def _candidate_position(item, matrix):
    az, alt = np.deg2rad([item['azimuth_deg'], item['altitude_deg']])
    ray = np.array([math.cos(alt)*math.cos(az), math.cos(alt)*math.sin(az), math.sin(alt)])
    return matrix[:3, :3] @ (ray*item['distance_m']) + matrix[:3, 3]


def prepare_ground_surface(lidar_world, *, query_regions=None):
    if lidar_world is None:
        return None
    cloud = np.asarray(lidar_world, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        return None
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    if len(cloud) < 8:
        return None
    if query_regions is not None:
        positions = []
        try:
            for observation, sensor_matrix in query_regions:
                matrix = np.asarray(sensor_matrix, dtype=float)
                if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                    raise ValueError('Invalid sensor matrix')
                positions.extend(_candidate_position(item, matrix)
                                 for item in observation.get('_ground_candidates', []))
            if positions:
                xy = np.asarray(positions)[:, :2]
                if not np.isfinite(xy).all():
                    raise ValueError('Invalid candidate coordinates')
                # Every exact-radius support ball lies inside this union AABB.
                # Keep a numeric margin; no plane support or obstacle is removed.
                lower, upper = xy.min(axis=0)-2.5-1e-8, xy.max(axis=0)+2.5+1e-8
                shape = np.floor((upper-lower)/2.5).astype(np.int64)+1
                if 0 < int(shape[0])*int(shape[1]) <= 1000000:
                    grid = np.zeros(tuple(shape),dtype=bool)
                    for point in xy:
                        first = np.floor((point-2.5-1e-8-lower)/2.5).astype(np.int64)
                        last = np.floor((point+2.5+1e-8-lower)/2.5).astype(np.int64)
                        grid[first[0]:last[0]+1,first[1]:last[1]+1] = True
                    cells = np.floor((cloud[:, :2]-lower)/2.5).astype(np.int64)
                    inside = ((cells >= 0) & (cells < shape)).all(axis=1)
                    keep = np.zeros(len(cloud),dtype=bool)
                    valid = cells[inside]
                    keep[inside] = grid[valid[:,0],valid[:,1]]
                    cloud = cloud[keep]
        except (KeyError, TypeError, ValueError, OverflowError):
            pass
    return cloud, cKDTree(cloud[:, :2],leafsize=32,balanced_tree=False,compact_nodes=False)


def _fit_local_plane(a,z,point,matrix):
    """Use a small well-conditioned solve; retain SVD at decision boundaries."""
    gram=a.T@a
    try:
        eigenvalues=np.linalg.eigvalsh(gram)
        if eigenvalues[0]<=0:raise np.linalg.LinAlgError('Rank deficient plane')
        singular=np.sqrt(eigenvalues[::-1])
        condition=singular[0]/singular[-1]
        if condition>100.:
            raise np.linalg.LinAlgError('Ill-conditioned plane uses reference SVD')
        fit=np.linalg.solve(gram,a.T@z)
        residual=a@fit-z
        origin=fit[2]+fit[:2]@(matrix[:2,3]-point[:2])
        margins=(abs(condition-100.),abs(np.linalg.norm(fit[:2])-.25),
            abs(np.sqrt(np.mean(residual**2))-.015),abs(np.max(np.abs(residual))-.04),
            abs(matrix[2,3]-origin-.5),abs(matrix[2,3]-origin-2.5),abs(abs(point[2]-fit[2])-.02))
        if min(margins)<=1e-8:raise np.linalg.LinAlgError('Threshold boundary uses reference SVD')
        return fit,3,singular
    except np.linalg.LinAlgError:
        fit,_,rank,singular=np.linalg.lstsq(a,z,rcond=None)
        return fit,rank,singular


def _impossible_surface_plane(xy,z):
    """Necessary pairwise bound for the unchanged slope/residual thresholds."""
    low,high=int(np.argmin(z)),int(np.argmax(z))
    gap=float(z[high]-z[low])
    separation=float(np.linalg.norm(xy[high]-xy[low]))
    # A slope <= .25 and maximum residual <= .04 imply this bound.
    # Near the boundary the original full fit still decides.
    return gap>.25*separation+.08+1e-8


def filter_ground(observation, lidar_world, sensor_matrix, *, surface=None):
    result = deepcopy(observation)
    points = result.pop('_ground_candidates', [])
    result.pop('_sensor_matrix', None)
    result.pop('_filtered', None)
    diagnostics = dict(status='unavailable', rejected=0, examined=0, retained_reasons={})
    result['ground_filter'] = diagnostics
    if (lidar_world is None and surface is None) or sensor_matrix is None:
        return result
    matrix = np.asarray(sensor_matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return result
    if surface is None:
        surface = prepare_ground_surface(lidar_world)
    if surface is None:
        return result
    cloud, tree = surface
    diagnostics['status'] = 'same_frame_local_plane'

    def retain(reason):
        counts=diagnostics['retained_reasons'];counts[reason]=counts.get(reason,0)+1
        return False

    positions = [_candidate_position(item, matrix) for item in points]
    # One exact-radius query batch replaces per-return Python/C transitions.
    # Preserve unsorted KD-tree traversal and the original support radius.
    neighborhoods = (tree.query_ball_point(np.asarray(positions)[:, :2], 2.5,
        return_sorted=False) if positions else [])

    def is_ground(point, indices):
        diagnostics['examined'] += 1
        near = cloud[indices]
        if len(near) < 8:
            return retain('insufficient_support')
        xy = near[:, :2]-point[:2]
        # Require spatial support on both sides, not extrapolation from a wall.
        if not ((xy.min(axis=0) < -.25).all() and (xy.max(axis=0) > .25).all()):
            return retain('extrapolation')
        if _impossible_surface_plane(xy,near[:,2]):
            return retain('nonplanar_or_ill_conditioned')
        a = np.column_stack([xy, np.ones(len(near))])
        fit, rank, singular = _fit_local_plane(a,near[:,2],point,matrix)
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

    from radar_support_precheck import screen_support
    screened = screen_support(cloud, positions, neighborhoods, matrix)
    kept = []
    for index, (item, point, indices) in enumerate(zip(points, positions, neighborhoods)):
        code = 0 if screened is None else int(screened[index])
        if code:
            diagnostics['examined'] += 1
            if code == 6:
                diagnostics['rejected'] += 1
            else:
                retain({1: 'insufficient_support', 2: 'extrapolation',
                    3: 'nonplanar_or_ill_conditioned', 4: 'surface_above_ground_band',
                    5: 'above_surface'}[code])
                kept.append(item)
            continue
        if is_ground(point, indices):
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
