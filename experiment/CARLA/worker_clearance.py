"""Physical worker clearance relative to the configured ego route lane."""
import math


def worker_trigger_ready(anchor_m, ego_route_m, event):
    distance=float(event.get('safety',{}).get('minimum_trigger_distance_m',75))
    gap=float(anchor_m)-float(ego_route_m)
    if not math.isfinite(distance) or distance<=0 or not math.isfinite(gap):
        raise RuntimeError('invalid worker trigger distance')
    if gap<=0:
        raise RuntimeError('worker crossing trigger was missed')
    return gap<=distance


def worker_clear_of_lane(worker, waypoint, clearance_m=.2, target_location=None):
    pose=worker.get_transform()
    lane_pose=waypoint.transform
    width=float(waypoint.lane_width)
    if not math.isfinite(width) or width<=0:
        raise ValueError('invalid worker crossing lane width')
    yaw=math.radians(lane_pose.rotation.yaw)
    actor_yaw=math.radians(pose.rotation.yaw)
    box=worker.bounding_box
    box_yaw=actor_yaw+math.radians(box.rotation.yaw)
    center_x=pose.location.x+math.cos(actor_yaw)*box.location.x-math.sin(actor_yaw)*box.location.y
    center_y=pose.location.y+math.sin(actor_yaw)*box.location.x+math.cos(actor_yaw)*box.location.y
    lateral=-(center_x-lane_pose.location.x)*math.sin(yaw)+(center_y-lane_pose.location.y)*math.cos(yaw)
    radius=abs(math.sin(box_yaw-yaw))*box.extent.x+abs(math.cos(box_yaw-yaw))*box.extent.y
    if target_location is not None:
        lateral=-(target_location.x-lane_pose.location.x)*math.sin(yaw)+(target_location.y-lane_pose.location.y)*math.cos(yaw)
        # Future walker heading may differ; use an enclosing radius for planning.
        radius=math.hypot(box.extent.x,box.extent.y)+math.hypot(box.location.x,box.location.y)
    margin=abs(lateral)-radius-width/2
    if not all(math.isfinite(value) for value in (margin,clearance_m)) or clearance_m<0:
        raise ValueError('invalid worker clearance geometry')
    return dict(clear=margin>=clearance_m,body_clearance_m=margin,required_clearance_m=clearance_m,
                geometry='planned_heading_independent_radius' if target_location is not None else 'observed_oriented_box')
