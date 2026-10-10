"""Measured cross-view pedestrian positions; no scene actor or event access."""
import math
import numpy as np


def clothing_descriptor(rgb, box):
    image = np.asarray(rgb)
    x1, y1, x2, y2 = map(float, box)
    height, width = image.shape[:2]
    if not 2 < x1 < x2 < width - 2 or not 2 < y1 < y2 < height - 2:
        return None
    crop = image[int(y1 + .2 * (y2-y1)):int(y1 + .65 * (y2-y1)),
                 int(x1 + .25 * (x2-x1)):int(x2 - .25 * (x2-x1))]
    if crop.size < 72:
        return None
    colors = crop.reshape(-1, 3).astype(float)
    colors = colors[colors.sum(axis=1) > 45]
    if len(colors) < 24:
        return None
    chroma = colors / colors.sum(axis=1, keepdims=True)
    histogram, _ = np.histogramdd(chroma[:, :2], bins=(12, 12), range=((0, 1), (0, 1)))
    import cv2
    histogram = cv2.GaussianBlur(histogram, (3, 3), .7).ravel()
    norm = np.linalg.norm(histogram)
    return histogram / norm if norm > 0 else None


def measured_pedestrian_point(cloud, pose, size, fov, box, road):
    from .passing_conditions import lift_box_body
    if cloud is None or not isinstance(fov, (float, int)) or not math.isfinite(fov):
        return None
    body = lift_box_body(cloud, pose, size, fov, box)
    if body is None:
        return None
    route = np.asarray(road, dtype=float)
    if route.ndim != 2 or route.shape[1] != 3 or len(route) < 2:
        return None
    center = np.median(body, axis=0)
    delta = np.diff(route[:, :2], axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    if not np.isfinite(route).all() or (lengths < .01).any():
        return None
    fraction = np.clip(((center[:2]-route[:-1, :2])*delta).sum(axis=1)/lengths**2, 0, 1)
    distance = np.linalg.norm(center[:2]-route[:-1, :2]-fraction[:, None]*delta, axis=1)
    index = int(np.argmin(distance))
    floor = route[index, 2] + fraction[index]*(route[index+1, 2]-route[index, 2])
    elevated = body[(body[:, 2]-floor > .35) & (body[:, 2]-floor < 2.8)]
    if len(elevated) < 4 or distance[index] > 20:
        return None
    extent = np.quantile(elevated[:, :2], .9, axis=0)-np.quantile(elevated[:, :2], .1, axis=0)
    if max(extent) > 1.2:
        return None
    # Identity altitude follows the supported road level; XY is measured, not a ray guess.
    return [*np.median(elevated[:, :2], axis=0).tolist(), float(floor)]


def merge_cross_view_observations(observations):
    """Fuse complete, distinct-view cliques; crowded or inconsistent groups stay separate."""
    candidates = {}
    for i, first in enumerate(observations):
        for j in range(i + 1, len(observations)):
            second = observations[j]
            if first.get('view', 'front') == second.get('view', 'front'):
                continue
            a, b = first.get('appearance'), second.get('appearance')
            if a is None or b is None or float(np.asarray(a) @ b) < .96:
                continue
            if np.linalg.norm(np.asarray(first['position'])-second['position']) > .65:
                continue
            candidates.setdefault(i, []).append(j)
            candidates.setdefault(j, []).append(i)
    unvisited = set(range(len(observations)))
    result = []
    while unvisited:
        component = {min(unvisited)}
        pending = list(component)
        while pending:
            index = pending.pop()
            for neighbor in candidates.get(index, []):
                if neighbor not in component:
                    component.add(neighbor)
                    pending.append(neighbor)
        unvisited -= component
        ordered = sorted(component)
        views = [observations[index].get('view', 'front') for index in ordered]
        unique = len(set(views)) == len(views)
        complete = all(component-{index} <= set(candidates.get(index, [])) for index in ordered)
        groups = [[observations[index] for index in ordered]] if unique and complete else [
            [observations[index]] for index in ordered]
        for group in groups:
            primary = max(group, key=lambda item: item.get('age_frames', 0))
            item = dict(primary)
            item['aliases'] = [entry['track_id'] for entry in group]
            item['position'] = np.median([entry['position'] for entry in group], axis=0).tolist()
            result.append(item)
    return result
