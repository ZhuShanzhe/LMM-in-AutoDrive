"""Bounded occlusion recovery; recovered identity is not completion evidence."""
import math
import numpy as np


def body_appearance(rgb, points, inverse_camera, fov):
    image = np.asarray(rgb)
    body = np.asarray(points, dtype=float)
    matrix = np.asarray(inverse_camera, dtype=float)
    if image.ndim != 3 or image.shape[2] != 3 or matrix.shape != (4, 4):
        return None
    if body.ndim != 2 or body.shape[1] != 3 or len(body) < 4:
        return None
    if not np.isfinite(body).all() or not np.isfinite(matrix).all():
        return None
    if not isinstance(fov, (int, float)) or not math.isfinite(fov) or not 10 < fov < 170:
        return None
    camera = body @ matrix[:3, :3].T + matrix[:3, 3]
    depth = camera[:, 0]
    height, width = image.shape[:2]
    focal = width / (2 * math.tan(math.radians(fov) / 2))
    safe = np.maximum(depth, 1e-6)
    u = np.rint(width / 2 + focal * camera[:, 1] / safe).astype(int)
    v = np.rint(height / 2 - focal * camera[:, 2] / safe).astype(int)
    valid = (depth > .5) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    pixels = np.unique(np.column_stack([v[valid], u[valid]]), axis=0)
    if len(pixels) < 4:
        return None
    offsets = np.array([(y, x) for y in (-1, 0, 1) for x in (-1, 0, 1)])
    pixels = np.unique((pixels[:, None, :] + offsets).reshape(-1, 2), axis=0)
    pixels = pixels[(pixels[:, 0] >= 0) & (pixels[:, 0] < height)
                    & (pixels[:, 1] >= 0) & (pixels[:, 1] < width)]
    if len(pixels) < 24:
        return None
    colors = image[pixels[:, 0], pixels[:, 1]].astype(float)
    brightness = colors.sum(axis=1)
    colors = colors[brightness > 45]
    if len(colors) < 24:
        return None
    chroma = colors / colors.sum(axis=1, keepdims=True)
    histogram, _ = np.histogramdd(chroma[:, :2], bins=(12, 12), range=((0, 1), (0, 1)))
    # Smoothing tolerates exposure and raster changes, not arbitrary colors.
    import cv2
    histogram = cv2.GaussianBlur(histogram, (3, 3), .7).ravel()
    norm = np.linalg.norm(histogram)
    return histogram / norm if norm > 0 else None


def road_coordinates(point, road):
    route = np.asarray(road, dtype=float)
    if route.ndim != 2 or route.shape[1] < 2 or len(route) < 3 or not np.isfinite(route).all():
        return None
    delta = np.diff(route[:, :2], axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    if (lengths < .01).any():
        return None
    point = np.asarray(point)[:2]
    fraction = np.clip(((point - route[:-1, :2]) * delta).sum(axis=1) / lengths**2, 0, 1)
    projected = route[:-1, :2] + fraction[:, None] * delta
    distances = np.linalg.norm(projected - point, axis=1)
    index = int(np.argmin(distances))
    if distances[index] > 6 or (index == 0 and fraction[index] <= 0):
        return None
    tangent = delta[index] / lengths[index]
    offsets = np.r_[0., np.cumsum(lengths)]
    lateral = float((point - projected[index]) @ np.array([-tangent[1], tangent[0]]))
    return route, lengths, offsets, index, fraction[index], tangent, lateral


def road_prediction(track, road, age):
    coordinates = road_coordinates(track['position'], road)
    if coordinates is None:
        return None
    route, lengths, offsets, index, fraction, tangent, lateral = coordinates
    delta = np.diff(route[:, :2], axis=0)
    speed = float(np.asarray(track['velocity']) @ tangent)
    if not .5 < speed < 20:
        return None
    coordinate = offsets[index] + fraction * lengths[index] + speed * age
    if not 0 < coordinate < offsets[-1]:
        return None
    new_index = min(int(np.searchsorted(offsets, coordinate, side='right') - 1), len(lengths) - 1)
    new_tangent = delta[new_index] / lengths[new_index]
    center = route[new_index, :2] + (coordinate - offsets[new_index]) * new_tangent
    return center + lateral * np.array([-new_tangent[1], new_tangent[0]]), new_tangent


def occlusion_score(track, measurement, timestamp_s, road):
    age = timestamp_s - track['last_seen_s']
    history = track.get('history', ())
    descriptors = track.get('appearances', ())
    appearance = measurement.get('appearance')
    mature = (track.get('motion_history_established') is True or
              len(history) >= 8 and history[-1][0] - history[0][0] >= .5)
    cyclist=track['category']=='cyclist'
    compatible=(track['category']==measurement['category'] or cyclist
        and measurement['category'] in {'motorcycle','pedestrian'})
    if (not (.5 if cyclist else 2.) < age <= 8 or not compatible
            or measurement.get('box_cropped') or not track.get('velocity_valid')
            or not mature
            or appearance is None or not descriptors):
        return None
    if cyclist:
        body=np.asarray(measurement.get('surface',[]))
        if (body.ndim!=2 or body.shape[1]!=3 or len(body)<4
                or not np.isfinite(body).all()
                or not 2.<=np.linalg.norm(track['velocity'])<=15.):
            return None
    similarity = max(float(np.asarray(old) @ appearance) for old in descriptors)
    if not math.isfinite(similarity) or similarity < .94:
        return None
    predicted = road_prediction(track, road, age)
    if predicted is None:
        return None
    point, tangent = predicted
    difference = np.asarray(measurement['position'])[:2] - point
    along = abs(float(difference @ tangent))
    across = abs(float(difference @ np.array([-tangent[1], tangent[0]])))
    # A previously steady slow vehicle can accelerate while occluded. Keep
    # lateral/appearance/uniqueness gates strict; widen only reachable distance.
    along_bound = min(14., 3. + .3 * age**2)
    if along > along_bound or across > 1.25:
        return None
    return along / along_bound + across / 1.25 + 4 * (1 - similarity)


def bound_cyclist_handover_score(track, candidate, measurement, timestamp_s, road):
    """Merge a mature measured replacement track, never a single-frame guess."""
    if candidate is None or candidate is track:
        return None
    age = timestamp_s - track['last_seen_s']
    motion = track.get('validated_motion') or {}
    motion_age = timestamp_s - motion.get('observed_at_s', -float('inf'))
    body = np.asarray(measurement.get('surface', []))
    history = [sample for sample in candidate.get('history', ())
               if 0. <= timestamp_s-sample[0] <= 1.]
    appearance = measurement.get('appearance')
    descriptors = track.get('appearances', ())
    if (track['category'] != 'cyclist' or candidate['category'] != 'cyclist'
            or measurement['category'] not in {'cyclist', 'motorcycle', 'pedestrian'}
            or not .5 < age <= 30. or not 0. <= motion_age <= 30.
            or not track.get('motion_history_established')
            or not candidate.get('velocity_valid')
            or not 0. <= timestamp_s-candidate['last_seen_s'] <= .25
            or len(history) < 8 or history[-1][0]-history[0][0] < .5
            or measurement.get('box_cropped') or body.ndim != 2
            or body.shape[1] != 3 or len(body) < 4 or not np.isfinite(body).all()
            or appearance is None or not descriptors):
        return None
    times=np.asarray([sample[0] for sample in history],dtype=float)
    points=np.asarray([sample[1] for sample in history],dtype=float)
    old_position=np.asarray(track['position'],dtype=float)
    current_position=np.asarray(measurement['position'],dtype=float)
    if (not np.isfinite(times).all() or not np.isfinite(points).all()
            or np.any(np.diff(times)<=0.) or np.any(np.diff(times)>.25+1e-8)
            or old_position.ndim!=1 or len(old_position)<2
            or current_position.ndim!=1 or len(current_position)<2
            or not np.isfinite(old_position).all() or not np.isfinite(current_position).all()):
        return None
    velocity = np.asarray(motion.get('velocity', []), dtype=float)
    candidate_velocity = np.asarray(candidate['velocity'], dtype=float)
    if (velocity.shape != (2,) or candidate_velocity.shape != (2,)
            or not np.isfinite(velocity).all() or not np.isfinite(candidate_velocity).all()
            or not 2. <= np.linalg.norm(velocity) <= 15.
            or not 2. <= np.linalg.norm(candidate_velocity) <= 15.):
        return None
    similarity = max(float(np.asarray(old) @ appearance) for old in descriptors)
    if not math.isfinite(similarity) or similarity < .96:
        return None
    prediction = road_prediction(dict(track, velocity=velocity), road, age)
    if prediction is None:
        return None
    point, tangent = prediction
    speed = float(candidate_velocity @ tangent)
    if (speed/np.linalg.norm(candidate_velocity) < .9
            or abs(speed-float(velocity @ tangent)) > 2.):
        return None
    delta = np.asarray(measurement['position'])[:2]-point
    along = abs(float(delta @ tangent))
    across = abs(float(delta @ np.array([-tangent[1], tangent[0]])))
    along_bound = min(14., 3. + .3*age**2)
    if along > along_bound or across > 1.25:
        return None
    return along/along_bound + across/1.25 + 4.*(1.-similarity)
