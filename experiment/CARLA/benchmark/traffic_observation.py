"""Snapshot-based traffic density evidence, separate from policy inputs."""
from collections import Counter
import math


def observe_traffic(snapshot, ego_id, actor_ids, world_map, max_distance_m=120, horizontal_fov_deg=110):
    ego=snapshot.find(ego_id)
    if ego is None:
        raise ValueError('ego missing from traffic observation frame')
    pose=ego.get_transform()
    origin=pose.location
    yaw=math.radians(pose.rotation.yaw)
    ego_lane=world_map.get_waypoint(origin)
    def key(waypoint):
        return (waypoint.road_id,waypoint.section_id,waypoint.lane_id) if waypoint else None
    counts=Counter()
    lanes=Counter()
    missing=0
    for identity in set(actor_ids)-{ego_id}:
        actor=snapshot.find(identity)
        if actor is None:
            missing+=1
            continue
        other=actor.get_transform()
        dx,dy=other.location.x-origin.x,other.location.y-origin.y
        distance=math.hypot(dx,dy)
        ahead=dx*math.cos(yaw)+dy*math.sin(yaw)
        side=-dx*math.sin(yaw)+dy*math.cos(yaw)
        if distance<3 or distance>max_distance_m or ahead<=0:
            continue
        if abs(math.atan2(side,ahead))>math.radians(horizontal_fov_deg/2):
            continue
        counts['front_cone']+=1
        alignment=math.cos(math.radians(other.rotation.yaw-pose.rotation.yaw))
        direction='same_direction' if alignment>.5 else 'opposite_direction' if alignment<-.5 else 'cross_direction'
        counts[direction]+=1
        lane=key(world_map.get_waypoint(other.location))
        if lane is not None:
            lanes[':'.join(map(str,lane))]+=1
        if direction=='same_direction' and lane is not None and lane==key(ego_lane):
            counts['ego_lane']+=1
    return dict(frame=snapshot.frame,counts={name:counts[name] for name in
        ('front_cone','same_direction','opposite_direction','cross_direction','ego_lane')},
        lane_counts=dict(lanes),actors_missing_from_snapshot=missing,
        scope='geometric_front_cone; no occlusion, illumination or camera detection check')


class TrafficDensitySummary:
    def __init__(self):
        self.frames=0
        self.total=Counter()
        self.minimum={}
        self.maximum=Counter()
        self.low_frames=0
        self.ego_empty_frames=0

    def update(self, observation):
        counts=observation['counts']
        self.frames+=1
        for key,value in counts.items():
            self.total[key]+=value
            self.minimum[key]=min(self.minimum.get(key,value),value)
            self.maximum[key]=max(self.maximum[key],value)
        self.low_frames+=counts['front_cone']<3
        self.ego_empty_frames+=counts['ego_lane']==0

    def result(self):
        return dict(frames=self.frames,minimum=self.minimum,maximum=dict(self.maximum),
            mean={key:value/self.frames for key,value in self.total.items()} if self.frames else {},
            below_three_front_actor_fraction=self.low_frames/self.frames if self.frames else None,
            empty_ego_lane_fraction=self.ego_empty_frames/self.frames if self.frames else None,
            scope='sampled geometry only; not visual traffic acceptance')
