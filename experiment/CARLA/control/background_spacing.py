"""Scenario traffic policy; never used as an ego perception or decision input."""
import math


def maintain_rear_spacing(manager, ego, actors, minimum_distance_m):
    if not isinstance(minimum_distance_m,(int,float)) or not math.isfinite(minimum_distance_m) or minimum_distance_m<=0.:
        return 0
    pose=ego.get_transform();origin=pose.location;direction=pose.get_forward_vector()
    changed=0
    for actor in actors:
        if not actor.is_alive or actor.id==ego.id:
            continue
        transform=actor.get_transform();location=transform.location
        forward=transform.get_forward_vector()
        dx,dy=location.x-origin.x,location.y-origin.y
        longitudinal=dx*direction.x+dy*direction.y
        lateral=abs(dx*direction.y-dy*direction.x)
        aligned=direction.x*forward.x+direction.y*forward.y
        if longitudinal<0. and lateral<=8. and aligned>.7 and math.hypot(dx,dy)<=150.:
            manager.distance_to_leading_vehicle(actor,float(minimum_distance_m))
            changed+=1
    return changed
