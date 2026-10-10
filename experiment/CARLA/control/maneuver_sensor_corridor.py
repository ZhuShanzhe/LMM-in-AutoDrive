"""Measured-pose swept path to a latched, static-map lane-change target."""
import math


def maneuver_sensor_corridor(target_points, ego_xyz, *, transition_m=10., horizon_m=20.):
    if (not isinstance(target_points, (list, tuple)) or len(target_points) < 3
            or not 5. <= transition_m <= 20. or not 5. <= horizon_m <= 85.):
        return None
    points = [tuple(point) for point in target_points]
    if any(len(point) != 3 or any(not isinstance(v, (int, float)) or isinstance(v, bool)
            or not math.isfinite(v) for v in point) for point in points + [tuple(ego_xyz)]):
        return None
    lengths = [math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(points, points[1:])]
    if any(length < .1 or abs(b[2]-a[2])/length > .3
           for a, b, length in zip(points, points[1:], lengths)):
        return None
    best = None; offset = 0.
    for a, b, length in zip(points, points[1:], lengths):
        dx, dy = b[0]-a[0], b[1]-a[1]
        fraction = max(0., min(1., ((ego_xyz[0]-a[0])*dx+(ego_xyz[1]-a[1])*dy)/length**2))
        projection = tuple(a[i]+fraction*(b[i]-a[i]) for i in range(3))
        distance = math.hypot(ego_xyz[0]-projection[0], ego_xyz[1]-projection[1])
        if best is None or distance < best[0]:
            best = (distance, offset+fraction*length, projection)
        offset += length
    if best[0] > 5. or offset-best[1] < 5.:
        return None
    shift = tuple(ego_xyz[i]-best[2][i] for i in range(3))
    result = []
    maximum = min(horizon_m, offset-best[1])
    samples = [float(index) for index in range(int(maximum)+1)]
    if maximum-samples[-1] > .01: samples.append(maximum)
    for sample in samples:
        target = best[1]+sample
        covered = 0.
        for a, b, length in zip(points, points[1:], lengths):
            if target <= covered+length+1e-8:
                fraction = max(0., min(1., (target-covered)/length))
                base = tuple(a[i]+fraction*(b[i]-a[i]) for i in range(3))
                blend = min(1., sample/transition_m)
                residual = 1.-(3.*blend**2-2.*blend**3)
                result.append(tuple(base[i]+residual*shift[i] for i in range(3)))
                break
            covered += length
    return result if len(result) >= 6 else None
