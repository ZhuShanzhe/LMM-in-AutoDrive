"""Fresh, directional sensor evidence shared by every lane-change gate."""

from copy import deepcopy
import math
import numpy as np
from .task_feasibility import predicate


def measured_lane_clearance(points, road, ego_location, *, half_width_m=1.65):
    """Verify swept-lane occupancy and positive road support in world XYZ."""
    result=dict(valid=False, occupied=None, observed=False,
                source='same_frame_lidar_target_lane_geometry')
    if points is None or road is None or ego_location is None:
        return dict(result, reason='target_lane_measurement_unavailable')
    xyz=np.asarray(points,dtype=float);path=np.asarray(road,dtype=float)
    if (xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)<20 or path.ndim!=2
            or path.shape[1]!=3 or len(path)<3 or not np.isfinite(xyz).all()
            or not np.isfinite(path).all()):
        return dict(result,reason='invalid_target_lane_geometry')
    lengths=np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1)
    if (lengths<.01).any() or (abs(np.diff(path[:,2]))/lengths>.3).any():
        return dict(result,reason='invalid_target_lane_surface')
    ego=np.asarray(ego_location,dtype=float)
    xyz=xyz[np.linalg.norm(xyz[:,:2]-ego[:2],axis=1)<65.]
    radius=half_width_m+1e-8
    lower=path[:,:2].min(axis=0)-radius;upper=path[:,:2].max(axis=0)+radius
    xyz=xyz[((xyz[:,:2]>=lower)&(xyz[:,:2]<=upper)).all(axis=1)]
    # Include ego projection to bind the measured horizon to current position.
    query=np.vstack([xyz,ego]);lateral=np.full(len(query),np.inf)
    progress=np.zeros(len(query));height=np.zeros(len(query));offset=0.
    for first,last,length in zip(path[:-1],path[1:],lengths):
        delta=last[:2]-first[:2]
        t=np.clip(((query[:,:2]-first[:2])*delta).sum(axis=1)/length**2,0.,1.)
        distance=np.linalg.norm(query[:,:2]-first[:2]-t[:,None]*delta,axis=1)
        take=distance<lateral
        lateral[take]=distance[take];progress[take]=offset+t[take]*length
        height[take]=query[take,2]-(first[2]+t[take]*(last[2]-first[2]))
        offset+=length
    signed=progress[:-1]-progress[-1];lateral=lateral[:-1];height=height[:-1]
    in_lane=(lateral<=half_width_m)&(signed>=-10.)&(signed<=20.)
    obstacle=in_lane&(height>.3)&(height<3.)
    # A measured road surface is required; a sparse/empty cloud is not clearance.
    ground=in_lane&(abs(height)<=.25)
    counts=[int((ground&(signed>=a)&(signed<a+5.)).sum()) for a in (-10.,-5.,0.,5.,10.,15.)]
    observed=all(count>=2 for count in counts)
    return dict(result,valid=True,occupied=bool(obstacle.any()),observed=observed,
        ground_support_counts=counts,obstacle_points=int(obstacle.sum()),
        front_obstacle_points=int((obstacle&(signed>=-3.)).sum()),
        rear_obstacle_points=int((obstacle&(signed<-3.)).sum()),
        rear_obstacle_nearest_m=float(abs(signed[obstacle&(signed<-3.)]).min())
            if (obstacle&(signed<-3.)).any() else None,
        nearest_obstacle_m=float(abs(signed[obstacle]).min()) if obstacle.any() else None,
        obstacle_samples=xyz[obstacle][:8].tolist(),
        reason='target_lane_occupied' if obstacle.any() else 'measured_target_lane_clear' if observed
        else 'target_lane_surface_not_fully_observed', rear_horizon_m=10.,front_horizon_m=20.)


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

    def update(self,risk,direction,visual,lidar_bev,*,timestamp_s,sensor_frame,frame_age_s,speed_mps,corridor_points=None,continuing=False,target_entered=False,physical_evidence=None,road_legal=None):
        result=deepcopy(risk)
        evidence=physical_evidence if physical_evidence is not None else lidar_lane_evidence(lidar_bev,direction,speed_mps,corridor_points)
        lane=(visual.get('lane_change') or {}).get(direction) or {}
        limited_completion=continuing and visual.get('risk_level')=='medium'
        task_specific=physical_evidence is not None
        visual_available=visual.get('source')!='unavailable_view' and 'requested_camera_unavailable' not in (visual.get('reason_codes') or [])
        visual_clear=(visual_available and evidence.get('observed') is True) if task_specific else (
            (visual.get('risk_level')=='low' and lane.get('is_safe') is True) or limited_completion)
        if task_specific:limited_completion=False
        fresh=math.isfinite(frame_age_s) and 0. <= frame_age_s <= self.max_age_s
        rear_distance=evidence.get('rear_obstacle_nearest_m')
        rear_only_completion=bool(task_specific and continuing and target_entered
            and evidence.get('occupied') is True and evidence.get('front_obstacle_points')==0
            and evidence.get('rear_obstacle_points')==evidence.get('obstacle_points')
            and isinstance(evidence.get('obstacle_points'),int) and evidence['obstacle_points']>0
            and isinstance(rear_distance,(int,float)) and not isinstance(rear_distance,bool)
            and math.isfinite(rear_distance) and rear_distance>=5.)
        clear=fresh and evidence['valid'] and (not evidence['occupied'] or rear_only_completion) and visual_clear
        if task_specific:clear=clear and road_legal is True
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
        if not evidence['valid']:reasons.append(evidence.get('reason','target_lane_measurement_unavailable'))
        elif evidence['occupied'] and not rear_only_completion:reasons.append('physical_target_lane_occupied')
        if not visual_clear:reasons.append('visual_target_lane_not_clear')
        if task_specific and road_legal is not True:
            reasons.append('lane_change_map_permission_unknown' if road_legal is None else 'lane_change_map_permission_denied')
        if clear and not stable:reasons.append('await_fresh_continuous_lane_clearance')
        speed_cap=8. if task_specific else 5. if limited_completion else None
        # A fully measured 20 m forward / 10 m rear corridor supports a
        # rolling merge; incomplete geometry retains the original crawl cap.
        horizons=[evidence.get('front_horizon_m'),evidence.get('rear_horizon_m')]
        if (task_specific and stable and not rear_only_completion
                and evidence.get('source')=='same_frame_lidar_target_lane_geometry'
                and all(isinstance(v,(int,float)) and not isinstance(v,bool)
                        and math.isfinite(v) for v in horizons)
                and horizons[0]>=20. and horizons[1]>=10.):
            speed_cap=15.
        result.setdefault('lane_change',{})[direction]=dict(is_safe=bool(stable),reason_codes=reasons,
            speed_cap_kmh=speed_cap)
        result['target_lane_contract']=dict(direction=direction,visual=deepcopy(visual),lidar=evidence,
            frame=sensor_frame,frame_age_s=frame_age_s,clear_since_s=since,is_safe=bool(stable),
            can_execute=bool(stable),reason_codes=reasons,limited_completion=limited_completion,
            rear_only_completion=rear_only_completion,
            road_legal=road_legal,
            learned_risk_used_for_authority=not task_specific,policy_safety_credit=False)
        result.setdefault('task_feasibility',{})['CHANGE_LANE_'+direction.upper()]=predicate(
            'CHANGE_LANE_'+direction.upper(),stable,known=fresh and evidence['valid'] and road_legal is not None
            and (evidence.get('observed') is True or evidence.get('occupied') is True) if task_specific
            else fresh and evidence['valid'],reasons=reasons,evidence=deepcopy(result['target_lane_contract']))
        return result


def enforce_requested_lane(decision,risk,direction):
    if direction not in ('left','right'):
        return deepcopy(decision),None
    if decision.get('deferred_lane_change') is True:
        from .safety_bridge import validate_runtime_control_decision
        from .task_feasibility import task_permission
        policy=risk.get('behavior_policy') or {}
        cap=policy.get('max_speed_kmh')
        if cap is None:cap=policy.get('speed_cap_kmh')
        if (not validate_runtime_control_decision(decision,risk)
                and policy.get('disposition') in {'CLEAR','CAUTIOUS_PROCEED'}
                and not policy.get('physical_stop') and not policy.get('absolute_stop_active')
                and policy.get('allow_positive_acceleration') is not False
                and isinstance(cap,(int,float)) and not isinstance(cap,bool)
                and math.isfinite(cap) and 0<decision['target_speed_kmh']<=cap
                and task_permission(risk,'PATH_CLEAR',frame_id=decision.get('frame_id'))):
            # Target-lane occupancy does not forbid verified current-lane motion.
            return deepcopy(decision),'deferred_current_lane_only'
    judgment=(risk.get('lane_change') or {}).get(direction) or {}
    if judgment.get('is_safe') is True:
        if judgment.get('speed_cap_kmh') is not None:
            result=deepcopy(decision)
            result['target_speed_kmh']=min(float(result.get('target_speed_kmh',0.)),judgment['speed_cap_kmh'])
            for key in ('longitudinal_sequence_schema','target_acceleration_mps2','sequence_valid_until_s'):
                result.pop(key,None)
            return result,('task_feasibility_bounded_lane_execution'
                if (risk.get('target_lane_contract') or {}).get('learned_risk_used_for_authority') is False
                else 'sensor_clear_low_speed_lane_completion')
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
