"""Bounded physical walking on connected sidewalks, not a task-event controller."""
from dataclasses import dataclass
import math


def sidewalk_path(anchor, spacing_m=4.0, hops=3):
    if anchor is None or 'Sidewalk' not in str(anchor.lane_type):
        return []

    def extend(method):
        points = []
        current = anchor
        for _ in range(hops):
            candidates = [point for point in getattr(current, method)(spacing_m)
                          if 'Sidewalk' in str(point.lane_type)]
            if len(candidates) != 1:
                break
            point = candidates[0]
            a, b = current.transform.location, point.transform.location
            distance = math.hypot(a.x-b.x, a.y-b.y)
            if not 0.5 <= distance <= spacing_m*1.5 or abs(a.z-b.z) > 1.0:
                break
            points.append(point)
            current = point
        return points

    return list(reversed(extend('previous'))) + [anchor] + extend('next')


@dataclass
class SidewalkPatrol:
    actor: object
    points: tuple
    speed_mps: float
    index: int
    direction: int = 1
    disabled: bool = False
    walked_m: float = 0.0
    previous_xy: tuple | None = None

    def update(self, carla):
        if not self.actor.is_alive:
            return
        location = self.actor.get_location()
        if self.previous_xy is not None:
            self.walked_m += math.hypot(location.x-self.previous_xy[0], location.y-self.previous_xy[1])
        self.previous_xy = (location.x, location.y)
        target = self.points[self.index]
        dx, dy = target.x-location.x, target.y-location.y
        distance = math.hypot(dx, dy)
        if distance < 0.6 and len(self.points) > 1:
            if not 0 <= self.index+self.direction < len(self.points):
                self.direction *= -1
            self.index += self.direction
            target = self.points[self.index]
            dx, dy = target.x-location.x, target.y-location.y
            distance = math.hypot(dx, dy)
        # A spawned/displaced walker may not resume by cutting across the road.
        on_path = min(self._segment_distance(location, a, b)
                      for a, b in zip(self.points, self.points[1:])) if len(self.points)>1 else distance
        self.disabled = self.disabled or on_path > 1.2 or abs(target.z-location.z) > 1.5
        speed = self.speed_mps if not self.disabled and distance > 0.1 else 0.0
        self.actor.apply_control(carla.WalkerControl(
            direction=carla.Vector3D(x=dx/max(distance, 0.1), y=dy/max(distance, 0.1), z=0.0),
            speed=speed, jump=False))

    @staticmethod
    def _segment_distance(p, a, b):
        dx, dy = b.x-a.x, b.y-a.y
        scale = max(dx*dx+dy*dy, 1e-9)
        t = min(1.0, max(0.0, ((p.x-a.x)*dx+(p.y-a.y)*dy)/scale))
        return math.hypot(p.x-a.x-t*dx, p.y-a.y-t*dy)
