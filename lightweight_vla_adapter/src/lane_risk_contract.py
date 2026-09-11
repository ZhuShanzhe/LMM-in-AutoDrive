"""Fresh, directional sensor evidence shared by every lane-change gate."""

from copy import deepcopy
import math
import numpy as np


def continued_lane_waypoint(previous,current,location):
    """Resolve only topology-linked successors, never an unrelated same lane id."""
    candidates=[candidate for distance in (1.,3.,5.,10.,20.,30.) for candidate in previous.next(distance)]
    candidates=[candidate for candidate in candidates
        if (candidate.road_id,candidate.section_id)==(current.road_id,current.section_id)]
    return min(candidates,key=lambda candidate:candidate.transform.location.distance(location)) if candidates else None


def lidar_lane_evidence(bev, direction, speed_mps, corridor_points=None):
    if direction not in ('left','right'):
        raise ValueError('Unknown lane direction')
    values = bev.detach().cpu().numpy() if hasattr(bev,'detach') else np.asarray(bev)
    if values.ndim == 4:
        values=values[0]
    if values.shape != (4,64,64) or not np.isfinite(values).all() or not values[0].any():
        return dict(valid=False,occupied=None,reason='lidar_unavailable')
    x=60.-np.arange(64)*80./63.
    y=np.arange(64)*60./63.-30.
    side=-y if direction=='left' else y
    # Grid half-cell inflation accounts for raster quantization and vehicle width.
    corridor=(x[:,None]>=-10.) & (x[:,None]<=max(15.,float(speed_mps)*3.+5.))
    corridor=corridor & (side[None,:]>=1.4) & (side[None,:]<=5.6)
    xx,yy=np.broadcast_arrays(x[:,None],y[None,:])
    if corridor_points is not None:
        pts=np.asarray(corridor_points,dtype=float)
        if pts.ndim!=2 or pts.shape[1]!=2 or len(pts)<2 or not np.isfinite(pts).all():
            return dict(valid=False,occupied=None,reason='target_lane_geometry_unavailable')
        distance=np.full(xx.shape,np.inf)
        for a,b in zip(pts[:-1],pts[1:]):
            dx,dy=b-a;length2=dx*dx+dy*dy
            if length2<1e-6:continue
            t=np.clip(((xx-a[0])*dx+(yy-a[1])*dy)/length2,0.,1.)
            distance=np.minimum(distance,np.hypot(xx-a[0]-t*dx,yy-a[1]-t*dy))
        corridor=distance<1.65
    ego_footprint=(abs(xx)<3.)&(abs(yy)<1.8)
    obstacle=(values[0]>.5) & (values[1]*5.>-1.6) & corridor & ~ego_footprint
    return dict(valid=True,occupied=bool(obstacle.any()),occupied_cells=int(obstacle.sum()),
        obstacle_samples=np.stack([xx[obstacle],yy[obstacle],(values[1]*5.)[obstacle]],axis=-1)[:8].tolist(),
        source='physical_lidar_bev',reason='target_lane_occupied' if obstacle.any() else 'no_obstacle_return_in_target_corridor')


class LaneRiskContract:
    def __init__(self, clear_duration_s=.6, max_age_s=.25):
        self.clear_duration_s=float(clear_duration_s)
        self.max_age_s=float(max_age_s)
        self.history={}

    def reset(self):
        self.history.clear()

    def update(self,risk,direction,visual,lidar_bev,*,timestamp_s,sensor_frame,frame_age_s,speed_mps,corridor_points=None,continuing=False):
        result=deepcopy(risk)
        evidence=lidar_lane_evidence(lidar_bev,direction,speed_mps,corridor_points)
        lane=(visual.get('lane_change') or {}).get(direction) or {}
        limited_completion=continuing and visual.get('risk_level')=='medium'
        visual_clear=(visual.get('risk_level')=='low' and lane.get('is_safe') is True) or limited_completion
        fresh=math.isfinite(frame_age_s) and 0. <= frame_age_s <= self.max_age_s
        clear=fresh and evidence['valid'] and not evidence['occupied'] and visual_clear
        prior=self.history.get(direction)
        if not clear:
            since=None
        elif prior and prior['clear'] and 0. <= timestamp_s-prior['time'] <= self.max_age_s:
            since=prior['since']
        else:
            since=timestamp_s
        repeated=bool(prior and sensor_frame<=prior['frame'])
        if not repeated:
            self.history[direction]=dict(clear=clear,since=since,time=timestamp_s,frame=sensor_frame)
        stable=clear and not repeated and since is not None and timestamp_s-since>=self.clear_duration_s
        reasons=[]
        if not fresh:reasons.append('stale_lane_observation')
        if not evidence['valid']:reasons.append(evidence['reason'])
        elif evidence['occupied']:reasons.append('physical_target_lane_occupied')
        if not visual_clear:reasons.append('visual_target_lane_not_clear')
        if clear and not stable:reasons.append('await_fresh_continuous_lane_clearance')
        result.setdefault('lane_change',{})[direction]=dict(is_safe=bool(stable),reason_codes=reasons,
            speed_cap_kmh=5. if limited_completion else None)
        result['target_lane_contract']=dict(direction=direction,visual=deepcopy(visual),lidar=evidence,
            frame=sensor_frame,frame_age_s=frame_age_s,clear_since_s=since,is_safe=bool(stable),
            reason_codes=reasons,limited_completion=limited_completion,policy_safety_credit=False)
        return result


def enforce_requested_lane(decision,risk,direction):
    if direction not in ('left','right'):
        return deepcopy(decision),None
    judgment=(risk.get('lane_change') or {}).get(direction) or {}
    if judgment.get('is_safe') is True:
        if judgment.get('speed_cap_kmh') is not None:
            result=deepcopy(decision)
            result['target_speed_kmh']=min(float(result.get('target_speed_kmh',0.)),judgment['speed_cap_kmh'])
            for key in ('longitudinal_sequence_schema','target_acceleration_mps2','sequence_valid_until_s'):
                result.pop(key,None)
            return result,'sensor_clear_low_speed_lane_completion'
        return deepcopy(decision),None
    result=deepcopy(decision)
    # Never clear a stronger emergency, and never fall back into the opposite lane.
    emergency=bool(result.get('emergency')) or result.get('action')=='emergency_brake'
    result.update(action='emergency_brake' if emergency else 'stop',target_speed_kmh=0.,
        target_lane=None,target_location=None,emergency=emergency,decision_status='BLOCKED',
        allow_positive_acceleration=False,reason='requested_lane_wait_for_sensor_clearance')
    result['blocked_reason_codes']=list(dict.fromkeys([*(result.get('blocked_reason_codes') or []),
        *(judgment.get('reason_codes') or ['requested_lane_evidence_missing'])]))
    for key in ('longitudinal_sequence_schema','target_acceleration_mps2','sequence_valid_until_s'):
        result.pop(key,None)
    return result,'requested_lane_wait_for_sensor_clearance'
