"""Passing completion from classified RGB boxes and same-frame LiDAR bodies.

Default passing requires a measured rear body. The explicit cyclist encounter
mode instead uses the user-defined time-window/no-contact event criterion.
"""
from collections import deque
import math
import numpy as np
from lightweight_vla_adapter.src.route_projection_compiled import project_points_to_route, projection_available
from control.observed_vehicle_reidentification import (body_appearance, occlusion_score,
    road_coordinates, road_prediction, bound_cyclist_handover_score)


def object_surface_points(points, road):
    xyz=np.asarray(points,dtype=float);route=np.asarray(road,dtype=float)
    if (xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)<20
            or route.ndim!=2 or route.shape[1]!=3 or len(route)<2
            or not np.isfinite(xyz).all() or not np.isfinite(route).all()):
        return np.empty((0,3))
    lengths=np.linalg.norm(np.diff(route[:,:2],axis=0),axis=1)
    if (lengths<.01).any() or (np.abs(np.diff(route[:,2]))/lengths>.3).any():
        return np.empty((0,3))
    # Any accepted return is within 8 m of a road segment and above its
    # interpolated height. Crop by conservative bounds before exact matching.
    lower=route.min(axis=0)+np.array([-8.,-8.,.35])-1e-6
    upper=route.max(axis=0)+np.array([8.,8.,3.8])+1e-6
    xyz=xyz[((xyz>=lower)&(xyz<=upper)).all(axis=1)]
    if not len(xyz):return np.empty((0,3))
    if projection_available():
        lateral, _, height = project_points_to_route(xyz, route, lengths)
    else:
        lateral=np.full(len(xyz),np.inf);height=np.zeros(len(xyz))
        for first,last,length in zip(route[:-1],route[1:],lengths):
            delta=last[:2]-first[:2]
            fraction=np.clip(((xyz[:,:2]-first[:2])*delta).sum(axis=1)/length**2,0.,1.)
            distance=np.linalg.norm(xyz[:,:2]-first[:2]-fraction[:,None]*delta,axis=1)
            update=distance<lateral;lateral[update]=distance[update]
            height[update]=xyz[update,2]-(first[2]+fraction[update]*(last[2]-first[2]))
    return xyz[(lateral<=8.)&(height>.35)&(height<3.8)]


def lift_box_body(points, inverse_camera, image_size, fov, box, *, allow_partial=False,
                  sparse_cyclist=False):
    """A partial box may maintain a track, but never proves rear completion."""
    if not isinstance(fov,(int,float)) or not math.isfinite(fov) or not 10<fov<170:return None
    width,height=image_size;x1,y1,x2,y2=map(float,box)
    if not all(math.isfinite(v) for v in (x1,y1,x2,y2)):return None
    if allow_partial:
        if not (0<=x1<x2<=width and 0<=y1<y2<=height):return None
    elif not (2<x1<x2<width-2 and 2<y1<y2<height-2):return None
    matrix=np.asarray(inverse_camera,dtype=float)
    if matrix.shape!=(4,4) or not np.isfinite(matrix).all():return None
    if (not np.allclose(matrix[3],[0.,0.,0.,1.],atol=1e-5)
            or not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-3)):return None
    cloud=np.asarray(points,dtype=float)
    if cloud.ndim!=2 or cloud.shape[1]!=3 or len(cloud)<(2 if sparse_cyclist else 3):return None
    camera=cloud@matrix[:3,:3].T+matrix[:3,3]
    depth=camera[:,0];focal=width/(2*math.tan(math.radians(fov)/2))
    denominator=np.maximum(depth,1e-6)
    u=width/2+focal*camera[:,1]/denominator
    v=height/2-focal*camera[:,2]/denominator
    margin=0. if sparse_cyclist else .1
    selected=(depth>.5)&(depth<60)&(u>x1+margin*(x2-x1))&(u<x2-margin*(x2-x1))
    selected&=(v>y1+margin*(y2-y1))&(v<y2-margin*(y2-y1))
    indices=np.flatnonzero(selected)
    if len(indices)<(2 if sparse_cyclist else 3):return None
    bins=np.floor(depth[indices]/.75).astype(int)
    unique,counts=np.unique(bins,return_counts=True)
    peak=unique[np.argmax(counts)]
    near_peak=np.abs(bins-peak)<=1
    body=cloud[indices[near_peak]]
    if len(body)<(2 if sparse_cyclist else 4):return None
    competing=[int(count) for key,count in zip(unique,counts) if abs(key-peak)>2]
    if competing and max(competing)>=max(counts)*.75:return None
    return body


def sparse_body_consensus(history, surface, timestamp_s, *, frames_required=3,
                          window_s=.2, max_step_m=.65, max_span_m=1.1):
    """Confirm sparse target support across consecutive world-frame LiDAR scans."""
    points=np.asarray(surface,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<2 or not np.isfinite(points).all():
        return [],False
    if history and not 0.<timestamp_s-history[-1][0]<=.1+1e-8:
        history=[]
    history=list(history)+[(float(timestamp_s),points.copy())]
    history=[item for item in history if 0.<=timestamp_s-item[0]<=window_s+1e-8]
    centers=[np.median(item[1][:,:2],axis=0) for item in history]
    consecutive=(len(history)>=frames_required and all(
        0.<history[index][0]-history[index-1][0]<=.1+1e-8
        and np.linalg.norm(centers[index]-centers[index-1])<=max_step_m
        for index in range(1,len(centers))))
    span=np.linalg.norm(centers[-1]-centers[0]) if len(centers)>=frames_required else float('inf')
    return history,bool(consecutive and span<=max_span_m)


def semantic_body_nms(detections, *, include_riders=False):
    """Suppress duplicate subtype boxes, not adjacent different road users."""
    import torch
    from torchvision.ops import batched_nms
    categories_allowed={'vehicle','cyclist','motorcycle'}
    if include_riders:categories_allowed.add('pedestrian')
    selected=[d for d in detections if d.category in categories_allowed]
    if not selected:return []
    boxes=torch.tensor([d.bbox_xyxy for d in selected],dtype=torch.float32)
    scores=torch.tensor([d.confidence for d in selected],dtype=torch.float32)
    categories=torch.tensor([{'vehicle':0,'cyclist':1,'motorcycle':1,'pedestrian':2}[d.category]
        for d in selected],dtype=torch.int64)
    keep=batched_nms(boxes,scores,categories,.6).tolist()
    return sorted([selected[index] for index in keep],
        key=lambda d:{'cyclist':0,'vehicle':1,'motorcycle':2,'pedestrian':3}[d.category])


def road_relative_reference(point,ego,heading,road):
    target=road_coordinates(point,road);origin=road_coordinates(ego,road)
    if target is None or origin is None:return None
    route,lengths,offsets,index,fraction,tangent,lateral=target
    if index==len(lengths)-1 and fraction>=1.-1e-8:return None
    if float(origin[5]@np.asarray(heading))<.5:return None
    projected=route[index,:2]+fraction*(route[index+1,:2]-route[index,:2])
    delta=np.diff(route[:,:2],axis=0)
    other_fraction=np.clip(((np.asarray(point)[:2]-route[:-1,:2])*delta).sum(axis=1)/lengths**2,0,1)
    others=route[:-1,:2]+other_fraction[:,None]*delta
    coordinate=offsets[index]+fraction*lengths[index]
    distance=float(np.linalg.norm(projected-np.asarray(point)[:2]))
    ambiguous=(np.linalg.norm(others-np.asarray(point)[:2],axis=1)<=distance+.25)&(
        np.abs(offsets[:-1]+other_fraction*lengths-coordinate)>15.)
    if ambiguous.any():return None
    origin_coordinate=origin[2][origin[3]]+origin[4]*origin[1][origin[3]]
    return dict(ahead_m=float(coordinate-origin_coordinate),lateral_m=float(lateral-origin[6]),
        tangent=tangent,source='measured_body_and_static_road_local_reference')


def short_curve_score(track, measurement, timestamp_s, road):
    age=timestamp_s-track['last_seen_s']
    body=np.asarray(measurement.get('surface',[]))
    descriptors=track.get('appearances',())
    appearance=measurement.get('appearance')
    mature=track.get('motion_history_established') or (
        len(track.get('history',()))>=8 and track['history'][-1][0]-track['history'][0][0]>=.5)
    if (not 0<age<=2. or track['category']!='vehicle' or measurement['category']!='vehicle'
            or not mature or not track.get('velocity_valid') or measurement.get('box_cropped')
            or appearance is None or not descriptors or body.ndim!=2 or body.shape[1]!=3
            or len(body)<4 or not np.isfinite(body).all()):return None
    similarity=max(float(np.asarray(old)@appearance) for old in descriptors)
    if not math.isfinite(similarity) or similarity<.96:return None
    prediction=road_prediction(track,road,age)
    if prediction is None:return None
    point,tangent=prediction;difference=np.asarray(measurement['position'])[:2]-point
    along=abs(float(difference@tangent));across=abs(float(difference@np.array([-tangent[1],tangent[0]])))
    if along>3. or across>1.25:return None
    return along/3.+across/1.25+4.*(1.-similarity)


class PassingMemory:
    def __init__(self, slow_speed_ratio=.7, slow_observation_s=1.2):
        if not isinstance(slow_speed_ratio,(int,float)) or isinstance(slow_speed_ratio,bool) or not 0<slow_speed_ratio<1:
            raise ValueError('slow_speed_ratio must lie strictly between zero and one')
        self.slow_speed_ratio=float(slow_speed_ratio)
        if not isinstance(slow_observation_s,(int,float)) or not 1.<=slow_observation_s<=3.:
            raise ValueError('slow observation must be between one and three seconds')
        self.slow_observation_s=float(slow_observation_s)
        self.tracks={};self.aliases={};self.bound={};self.facts={};self.counter=0
        self.last_frame=None;self.last_time=None

    @staticmethod
    def category(step):
        action=step.get('action');condition=(step.get('parameters') or {}).get('condition')
        if action=='OVERTAKE' and condition=='SLOW_VEHICLE':return 'vehicle'
        if action=='PASS_BY' and condition=='CYCLIST':return 'cyclist'
        return None

    @staticmethod
    def surface_association(track, measurement, timestamp_s, forward, lateral):
        age=timestamp_s-track['last_seen_s']
        subtype_continuation=(track['category']=='cyclist'
            and measurement['category'] in {'motorcycle','pedestrian'})
        if not 0<=age<=2. or (track['category']!=measurement['category'] and not subtype_continuation):return None
        predicted=track['position']+track['velocity']*age
        delta=np.asarray(measurement['position'])[:2]-predicted
        longitudinal=abs(float(delta@forward));across=abs(float(delta@lateral))
        if subtype_continuation:
            # A measured rider/body can retain a mature bicycle identity when
            # its visible subtype changes, but cannot create a bicycle identity.
            body=np.asarray(measurement.get('surface',[]))
            recovering=(track.get('recovery_pending') is True
                and 0<=timestamp_s-track.get('recovery_started_s',-float('inf'))<=.5
                and track.get('recovery_velocity_valid') is True)
            if (age>.5+1e-8 or not track.get('motion_history_established')
                    or not (track.get('velocity_valid') or recovering) or body.ndim!=2
                    or body.shape[1]!=3 or len(body)<4 or not np.isfinite(body).all()
                    or longitudinal>1.5 or across>1.):return None
            if measurement['category']=='pedestrian' and not 2.<=np.linalg.norm(track['velocity'])<=15.:
                return None
            if age>.25+1e-8 or recovering:
                appearance=measurement.get('appearance');descriptors=track.get('appearances',())
                if appearance is None or not descriptors:return None
                similarity=max(float(np.asarray(old)@appearance) for old in descriptors)
                if not math.isfinite(similarity) or similarity<.94:return None
            return longitudinal/1.5+across
        if longitudinal<=6. and across<=1.5:return longitudinal/6.+across/1.5
        # Surface centroids move as visibility changes. Only a fresh, cropped
        # handover with overlapping measured body support may exceed that gate.
        if not measurement.get('box_cropped') or age>.25 or longitudinal>6. or across>2.75:return None
        old=track.get('surface')
        new=measurement.get('surface')
        if old is None or new is None:return None
        old=np.asarray(old)[:,:2]+track['velocity']*age;new=np.asarray(new)[:,:2]
        if len(old)<4 or len(new)<4 or not np.isfinite(old).all() or not np.isfinite(new).all():return None
        for axis,limit in ((forward,9.),(lateral,3.)):
            a=np.quantile(old@axis,[.05,.95]);b=np.quantile(new@axis,[.05,.95])
            if max(a[1]-a[0],b[1]-b[0])>limit:return None
            if max(a[0],b[0])-min(a[1],b[1])>.6:return None
        return longitudinal/6.+across/2.75+.3

    def update(self, measurements, *, frame, timestamp_s, ego_xy, heading, road=None):
        if frame==self.last_frame:return
        if self.last_time is not None and timestamp_s<self.last_time:
            self.tracks.clear();self.aliases.clear();self.bound.clear();self.facts.clear()
        self.last_frame,self.last_time=frame,timestamp_s
        ego=np.asarray(ego_xy,dtype=float);forward=np.asarray(heading,dtype=float)
        lateral=np.array([-forward[1],forward[0]])
        grouped={}
        recovered=set()
        verified_handovers=set()
        bound_ids={value['target_id'] for value in self.bound.values() if value.get('target_id')}
        for m in measurements:
            p=np.asarray(m['position'],dtype=float)[:2];alias=(m['view'],m['local_id'])
            identity=self.aliases.get(alias)
            # A temporary world-track split must not overwrite a continuously
            # observed image identity of a bound referent. Require measured
            # body, historical alias, appearance, motion and unique ownership.
            anchors=[]
            replacement=self.tracks.get(identity)
            for key in bound_ids:
                old=self.tracks.get(key)
                if (old is None or key==identity or identity in bound_ids
                        or key in verified_handovers):continue
                handover=(bound_cyclist_handover_score(old,replacement,m,timestamp_s,road)
                    if road is not None else None)
                if handover is not None:
                    competing=any(bound_cyclist_handover_score(old,other,body,timestamp_s,road)
                        is not None for owner,other in self.tracks.items()
                        if owner!=identity and owner not in bound_ids
                        for body in other.get('measurements',()))
                    if not competing:
                        anchors.append((handover,key,'mature_measured_cyclist_handover'))
                    continue
                alias_time=(old.get('image_aliases') or {}).get(alias)
                appearance=m.get('appearance');descriptors=old.get('appearances',())
                body=np.asarray(m.get('surface',[]))
                if (alias_time is None or not 0<=timestamp_s-alias_time<=2.
                        or appearance is None or not descriptors or m.get('box_cropped')
                        or body.ndim!=2 or len(body)<4):continue
                similarity=max(float(np.asarray(value)@appearance) for value in descriptors)
                if similarity<.96:continue
                score=self.surface_association(old,m,timestamp_s,forward,lateral)
                if score is None and road is not None:
                    score=short_curve_score(old,m,timestamp_s,road)
                if score is not None:anchors.append((score,key,'measured_bound_image_alias_continuity'))
            anchors.sort()
            if len(anchors)==1:
                previous_identity=identity
                identity=anchors[0][1]
                if anchors[0][2]=='mature_measured_cyclist_handover':
                    old=self.tracks[identity]
                    old['reidentified_gap_s']=timestamp_s-old['last_seen_s']
                    old['history']=deque(replacement['history'],maxlen=80)
                    verified_handovers.add(identity)
                    for alias_key,owner in list(self.aliases.items()):
                        if owner==previous_identity:self.aliases[alias_key]=identity
                    if previous_identity in grouped:
                        grouped.setdefault(identity,[]).extend(grouped.pop(previous_identity))
                    self.tracks.pop(previous_identity,None)
                self.aliases[alias]=identity
                self.tracks[identity]['identity_repair_source']=anchors[0][2]
            track=self.tracks.get(identity)
            continuation_only=m['category'] in {'motorcycle','pedestrian'}
            if continuation_only and identity not in verified_handovers:
                # Re-run the uniqueness test even if the image tracker reused
                # an alias; subtype changes must not bypass ambiguous bodies.
                identity=None
            if track is not None and identity not in verified_handovers:
                if (self.surface_association(track,m,timestamp_s,forward,lateral) is None
                        or any(old['view']==m['view'] and old['local_id']!=m['local_id']
                               for old in grouped.get(identity,[]))
                        ):
                    identity=None
            if identity is None:
                choices=[]
                for key,t in self.tracks.items():
                    if any(old['view']==m['view'] and old['local_id']!=m['local_id']
                           for old in grouped.get(key,[])):continue
                    score=self.surface_association(t,m,timestamp_s,forward,lateral)
                    if score is None and road is not None:
                        score=short_curve_score(t,m,timestamp_s,road)
                        if score is None:score=occlusion_score(t,m,timestamp_s,road)
                    if score is not None:choices.append((score,key))
                choices.sort()
                if len(choices)>1 and choices[1][0]-choices[0][0]<.25:continue
                if choices:
                    identity=choices[0][1]
                    recovered_after=(.5 if self.tracks[identity]['category']=='cyclist' else 2.)
                    if timestamp_s-self.tracks[identity]['last_seen_s']>recovered_after:
                        recovered.add(identity)
                else:
                    if continuation_only:continue
                    self.counter+=1;identity=f'measured-body-{self.counter}'
                    self.tracks[identity]=dict(category=m['category'],position=p.copy(),
                        velocity=np.zeros(2),last_seen_s=timestamp_s,history=deque(maxlen=80))
                self.aliases[alias]=identity
            if self.tracks[identity]['category']!=m['category']:
                m=dict(m,observed_category=m['category'],category=self.tracks[identity]['category'],
                    semantic_identity_source='fresh_measured_bicycle_or_rider_continuation')
            grouped.setdefault(identity,[]).append(m)
        for identity,values in grouped.items():
            t=self.tracks[identity];position=np.median([v['position'][:2] for v in values],axis=0)
            aliases=t.setdefault('image_aliases',{})
            aliases.update({(value['view'],value['local_id']):timestamp_s for value in values})
            t['image_aliases']={alias:time for alias,time in aliases.items() if timestamp_s-time<=30.}
            if identity in recovered:
                previous=t['history']
                if len(previous)>=8 and previous[-1][0]-previous[0][0]>=.5:
                    t['motion_history_established']=True
                t['reidentified_gap_s']=timestamp_s-t['last_seen_s']
                if t['category']=='cyclist':
                    t.update(recovery_pending=True,recovery_started_s=timestamp_s,
                        recovery_velocity_valid=bool(t.get('velocity_valid')))
                t['history'].clear()
            appearances=t.setdefault('appearances',deque(maxlen=8))
            for value in values:
                if value.get('appearance') is not None:
                    appearances.append(value['appearance'])
            history=t['history'];history.append((timestamp_s,position.copy()))
            if len(history)>=8 and history[-1][0]-history[0][0]>=.5:
                t['motion_history_established']=True
            recent=[(time,p) for time,p in history if timestamp_s-time<=.8]
            t['velocity_valid']=False
            if len(recent)>=4 and recent[-1][0]-recent[0][0]>=.15:
                times=np.array([time for time,p in recent]);positions=np.array([p for time,p in recent])
                times=times-times.mean();denominator=float(times@times)
                velocity=times@(positions-positions.mean(axis=0))/max(denominator,1e-6)
                fitted=positions.mean(axis=0)+times[:,None]*velocity
                residual=float(np.sqrt(np.mean(np.sum((positions-fitted)**2,axis=1))))
                t['velocity_valid']=residual<=.75 and np.linalg.norm(velocity)<=40.
                if t['velocity_valid']:
                    t['velocity']=velocity
                    t['recovery_pending']=False
                    t['validated_motion']=dict(observed_at_s=timestamp_s,
                        velocity=velocity.copy(),fit_residual_m=residual)
            t.update(position=position,last_seen_s=timestamp_s,frame=frame,
                ahead_m=float(np.dot(position-ego,forward)),
                lateral_m=float(np.dot(position-ego,lateral)),measurements=values,
                surface=np.concatenate([v['surface'] for v in values]) if all('surface' in v for v in values) else None)
            t.pop('road_geometry',None)
            if road is not None:
                local=road_coordinates(position,road)
                if local is not None:
                    _,_,_,_,_,tangent,signed_lateral=local
                    t['road_geometry']=dict(lateral_m=signed_lateral,tangent=tangent)
                reference=road_relative_reference(position,ego,forward,road)
                if reference is not None:t['acquisition_geometry']=reference
                else:t.pop('acquisition_geometry',None)
        for identity,t in list(self.tracks.items()):
            if timestamp_s-t['last_seen_s']>30.:
                del self.tracks[identity]
        self.aliases={key:value for key,value in self.aliases.items() if value in self.tracks}

    def stable_slow_motion(self, track, timestamp_s, forward, cruise_speed_kmh):
        history=[(time,p) for time,p in track['history'] if timestamp_s-time<=self.slow_observation_s+.1]
        if len(history)<12 or history[-1][0]-history[0][0]<self.slow_observation_s-1e-6:
            return False
        times=np.array([time for time,p in history]);positions=np.array([p for time,p in history])
        if np.max(np.diff(times))>.25:return False
        times=times-times[-1];longitudinal=positions@forward
        coefficients=np.polyfit(times,longitudinal,2)
        acceleration=2.*coefficients[0]
        start_speed=2.*coefficients[0]*times[0]+coefficients[1];end_speed=coefficients[1]
        residual=float(np.sqrt(np.mean((np.polyval(coefficients,times)-longitudinal)**2)))
        return (abs(acceleration)<=.6 and residual<=.5
            and .5<min(start_speed,end_speed)
            and max(start_speed,end_speed)<=self.slow_speed_ratio*cruise_speed_kmh/3.6)

    def proofs(self, document, *, frame_id, frame, timestamp_s, observed_at_s,
               ego_xy, heading, ego_rear_m, corridor_clear, cruise_speed_kmh=30.,passing_origin_side=None,
               target_binding_allowed=True):
        result={};forward=np.asarray(heading,dtype=float);ego=np.asarray(ego_xy,dtype=float)
        fresh=(self.last_frame==frame and isinstance(observed_at_s,(int,float))
            and 0<=timestamp_s-observed_at_s<=.25)
        for step in document['intent']['steps']:
            category=self.category(step)
            condition=(step.get('parameters') or {}).get('condition')
            key=(document['request_id'],step['step_id'])
            measured_clearance=None
            confirm=step.get('action')=='CONFIRM' and condition=='CYCLIST_PASSED'
            if category is None and not confirm:continue
            if confirm:
                dependencies=step.get('depends_on') or []
                explicit_target=any(step.get(k) or (step.get('parameters') or {}).get(k)
                                    for k in ('target','target_ref'))
                candidates=[fact for fact in self.facts.values() if fact['category']=='cyclist'
                    and (not dependencies or fact['source_request_id']==document['request_id']
                         and fact['source_step_id'] in dependencies)
                    and fact['completed_at_s']<=timestamp_s]
                # A standalone confirmation refers to the latest completed
                # passing event, not to a target's predicted current position.
                binding=self.bound.get(key)
                if binding is None and candidates and not explicit_target:
                    binding=dict(fact=max(candidates,key=lambda x:x['completed_at_s']))
                    self.bound[key]=binding
                fact=binding.get('fact') if binding and not explicit_target else None
                satisfied=bool(fact and corridor_clear and fresh)
                target_id=fact['target_id'] if fact else None
                reason='observed_prior_cyclist_pass' if fact else 'cyclist_pass_fact_missing'
                valid=fresh
            else:
                binding=self.bound.get(key)
                if binding and category=='vehicle' and target_binding_allowed and key not in self.facts:
                    retained=self.tracks.get(binding['target_id'])
                    if retained is None or timestamp_s-retained['last_seen_s']>2.:
                        # An unissued generic maneuver may refresh a lost referent;
                        # issued maneuvers keep their original identity and cannot rebind.
                        self.bound.pop(key,None);binding=None
                if binding is None and fresh and (category!='vehicle' or target_binding_allowed):
                    candidates=[]
                    nearest_vehicle_ahead=None
                    if category=='vehicle':
                        foreground=[]
                        for t in self.tracks.values():
                            if t['frame']!=frame or t['category']!='vehicle':continue
                            if not any(not m.get('box_cropped',False) for m in t.get('measurements',[])):continue
                            ahead=t['ahead_m'];across=t['lateral_m']
                            origin_sign={'RIGHT':1.,'LEFT':-1.}.get(passing_origin_side)
                            in_source_lane=(abs(across)<=1.8 or origin_sign is not None
                                and 2.<=origin_sign*across<=4.5)
                            if ego_rear_m+4.<ahead<55. and in_source_lane:foreground.append(ahead)
                        nearest_vehicle_ahead=min(foreground) if foreground else None
                    for identity,t in self.tracks.items():
                        if t['frame']!=frame or t['category']!=category:continue
                        if not any(not m.get('box_cropped',False) for m in t.get('measurements',[])):continue
                        if category=='cyclist' and not any(
                            len(m.get('surface',()))>=4
                            or (m.get('body_support_mode')=='temporally_confirmed_sparse'
                                and m.get('body_support_frames',0)>=3)
                            for m in t.get('measurements',[])):
                            continue
                        reference=(t.get('acquisition_geometry') or t) if category=='cyclist' else t
                        if not ego_rear_m+4.<reference['ahead_m']<55. or abs(reference['lateral_m'])>4.5:continue
                        if category=='vehicle':
                            # Do not choose a distant qualifying vehicle through
                            # a nearer unqualified road vehicle or stopped queue.
                            if nearest_vehicle_ahead is not None and t['ahead_m']>nearest_vehicle_ahead+5.:
                                continue
                            if abs(t['lateral_m'])>1.8:
                                origin_sign={'RIGHT':1.,'LEFT':-1.}.get(passing_origin_side)
                                if origin_sign is None or not 2.<=origin_sign*t['lateral_m']<=4.5:continue
                            speed=float(np.dot(t['velocity'],forward))
                            lateral_speed=abs(float(np.dot(t['velocity'],[-forward[1],forward[0]])))
                            if (not t.get('velocity_valid')
                                    or not .5<speed<=self.slow_speed_ratio*cruise_speed_kmh/3.6
                                    or not self.stable_slow_motion(t,timestamp_s,forward,cruise_speed_kmh)
                                    or lateral_speed>max(1.5,.45*max(0.,speed))):continue
                        candidates.append((reference['ahead_m'],identity))
                    candidates.sort()
                    if candidates and (len(candidates)==1 or candidates[1][0]-candidates[0][0]>5.):
                        binding=dict(target_id=candidates[0][1],bound_at_s=timestamp_s,
                            binding_origin='measured_before_maneuver' if target_binding_allowed else 'adjacent_observation',
                            clear_since_s=None,last_frame=None,last_time=None)
                        self.bound[key]=binding
                target_id=binding['target_id'] if binding else None
                track=self.tracks.get(target_id)
                measured_clearance=None
                if (fresh and track and track.get('frame')==frame
                        and not track.get('recovery_pending')):
                    rear_measurements=[m for m in track['measurements'] if m['view']=='rear'
                        and not m.get('box_cropped',False)]
                    rear_dense=[m for m in rear_measurements if len(m.get('surface',()))>=4]
                    rear_sparse=[m for m in rear_measurements
                        if category=='cyclist' and len(m.get('surface',()))>=2
                        and m.get('body_support_mode')=='temporally_confirmed_sparse'
                        and m.get('body_support_frames',0)>=3]
                    rear=rear_dense+rear_sparse
                    if rear:
                        surfaces=np.concatenate([m['surface'][:,:2] for m in rear])
                        # A margin covers unmeasured bumper protrusion. The
                        # nearest current body return, not a centroid, is used.
                        measured_clearance=float(-np.max((surfaces-ego)@forward)-ego_rear_m-2.)
                        required_clearance=10. if rear_sparse and not rear_dense else 8.
                    else:
                        required_clearance=8.
                else:
                    required_clearance=8.
                current_pass=(measured_clearance is not None
                    and measured_clearance>=required_clearance and corridor_clear)
                if binding and fresh and binding['last_frame']!=frame:
                    continuous=(binding['last_time'] is not None
                        and 0<observed_at_s-binding['last_time']<=.25 and binding['last_frame']!=frame)
                    if not current_pass or not continuous:binding['clear_since_s']=None
                    if current_pass and binding['clear_since_s'] is None:binding['clear_since_s']=observed_at_s
                    binding['last_time']=observed_at_s;binding['last_frame']=frame
                    if current_pass and observed_at_s-binding['clear_since_s']>=.5:
                        self.facts.setdefault(key,dict(target_id=target_id,category=category,
                            completed_at_s=observed_at_s,completion_sensor_frame=frame,
                            source_request_id=document['request_id'],source_step_id=step['step_id'],
                            measured_rear_clearance_m=measured_clearance,
                            required_rear_clearance_m=required_clearance,
                            rear_body_support=('temporally_confirmed_sparse' if rear_sparse and not rear_dense
                                else 'dense_current_frame'),
                            source='rgb_classified_same_frame_lidar_rear_body',event_status='COMPLETED'))
                fact=self.facts.get(key)
                satisfied=bool(fact and fresh and corridor_clear)
                valid=fresh
                reason='observed_body_passed_with_rear_clearance' if satisfied else 'target_pass_not_observed'
                if not binding and category=='vehicle' and not target_binding_allowed:
                    reason='target_not_acquired_before_maneuver'
            result[step['step_id']]=dict(request_id=document['request_id'],step_id=step['step_id'],
                frame_id=frame_id,sensor_frame=frame,condition=condition,valid=valid,
                satisfied=satisfied,can_proceed=satisfied,observed_at_s=observed_at_s,
                valid_until_s=observed_at_s+.25 if fresh else None,status='CLEAR' if satisfied else 'UNKNOWN',
                target_id=target_id,completed_event_fact=fact,reason=reason,
                source='rgb_lidar_passing_observer',policy_truth_access=False)
            diagnostic_track=self.tracks.get(target_id)
            if (fresh and diagnostic_track and diagnostic_track.get('frame')==frame
                    and diagnostic_track.get('category') in {'vehicle','cyclist'}
                    and diagnostic_track.get('velocity_valid') and passing_origin_side):
                geometry=diagnostic_track.get('road_geometry')
                normal=(np.array([-geometry['tangent'][1],geometry['tangent'][0]])
                        if geometry else np.array([-forward[1],forward[0]]))
                result[step['step_id']]['motion_evidence']=dict(
                    schema_version='measured_adjacent_passing/1.0', frame_id=frame_id,
                    sensor_frame=frame, observed_at_s=observed_at_s, target_id=target_id,
                    lateral_m=float(geometry['lateral_m'] if geometry else diagnostic_track['lateral_m']),
                    transverse_speed_mps=float(np.dot(diagnostic_track['velocity'],normal)),
                    coordinate_frame='MEASURED_ROAD_LOCAL' if geometry else 'EGO_TANGENT',
                    target_category=diagnostic_track['category'],velocity_valid=True,
                    ego_tangent_lateral_m=float(diagnostic_track['lateral_m']),
                    corridor_clear=bool(corridor_clear), origin_side=passing_origin_side,
                    policy_truth_access=False)
            result[step['step_id']]['target_observation']=dict(
                scope='measured_target_diagnostic_not_completion_permission',
                bound_at_s=(binding or {}).get('bound_at_s'),
                binding_origin=(binding or {}).get('binding_origin'),
                new_binding_allowed=bool(target_binding_allowed),
                track_retained=diagnostic_track is not None,
                measured_in_current_frame=bool(diagnostic_track and diagnostic_track.get('frame')==frame),
                last_seen_s=(diagnostic_track or {}).get('last_seen_s'),
                ahead_m=(diagnostic_track or {}).get('ahead_m'),
                lateral_m=(diagnostic_track or {}).get('lateral_m'),
                position_world_m=(diagnostic_track['position'].tolist() if diagnostic_track else None),
                velocity_world_mps=(diagnostic_track['velocity'].tolist() if diagnostic_track else None),
                velocity_valid=(bool(diagnostic_track.get('velocity_valid')) if diagnostic_track else None),
                motion_history_established=(diagnostic_track or {}).get('motion_history_established',False),
                reidentified_gap_s=(diagnostic_track or {}).get('reidentified_gap_s'),
                identity_repair_source=(diagnostic_track or {}).get('identity_repair_source'),
                image_identity_aliases=[dict(view=alias[0],local_id=alias[1],last_seen_s=stamp)
                    for alias,stamp in ((diagnostic_track or {}).get('image_aliases') or {}).items()],
                current_body_support=[dict(view=m.get('view'),mode=m.get('body_support_mode','dense_current_frame'),
                    frames=m.get('body_support_frames',1),surface_returns=len(m.get('surface',())))
                    for m in ((diagnostic_track or {}).get('measurements',[])
                        if diagnostic_track and diagnostic_track.get('frame')==frame else [])],
                current_history_samples=len((diagnostic_track or {}).get('history',())),
                observed_categories=sorted({m.get('observed_category',m['category'])
                    for m in (diagnostic_track or {}).get('measurements',[])}),
                acquisition_geometry_valid=isinstance((diagnostic_track or {}).get('acquisition_geometry'),dict),
                road_ahead_m=((diagnostic_track or {}).get('acquisition_geometry') or {}).get('ahead_m'),
                road_lateral_m=((diagnostic_track or {}).get('acquisition_geometry') or {}).get('lateral_m'),
                measured_rear_clearance_m=measured_clearance,
                corridor_clear=bool(corridor_clear),
                sensor_fresh=bool(fresh),
                clearance_hold_since_s=(binding or {}).get('clear_since_s'))
        while len(self.facts)>128:self.facts.pop(next(iter(self.facts)))
        while len(self.bound)>128:self.bound.pop(next(iter(self.bound)))
        return result


def _depends_on_step(step, source_id, by_id):
    pending=list(step.get('depends_on') or []);visited=set()
    while pending:
        identity=pending.pop()
        if identity==source_id:return True
        if identity in visited:continue
        visited.add(identity)
        pending.extend((by_id.get(identity) or {}).get('depends_on') or [])
    return False


def _requires_passing_target(step, lane_step):
    category=PassingMemory.category(step)
    # A dependent PASS_BY:CYCLIST step is itself an explicit target contract;
    # the preceding lane-change node need not duplicate target_ref.
    return category in {'vehicle','cyclist'}


def pending_vehicle_passing_lane(steps, active_step_id):
    """Guard the next merge before advance() can transition into it this frame."""
    start=next((i for i,s in enumerate(steps) if s['step_id']==active_step_id),None)
    if start is None:return None
    by_id={s['step_id']:s for s in steps}
    lane=next((s for s in steps[start:] if s.get('action')=='CHANGE_LANE'),None)
    if lane is None:return None
    return lane['step_id'] if any(_requires_passing_target(s,lane)
        and _depends_on_step(s,lane['step_id'],by_id) for s in steps[start:]) else None


def passing_target_binding_allowed(steps, active_step_id, *, maneuver_started, origin_side, lane_contract=None):
    active=next((s for s in steps if s['step_id']==active_step_id),{})
    prepared=(active.get('action')=='OVERTAKE' or (lane_contract or {}).get('road_legal') is True
        and ((lane_contract or {}).get('departure_window') or {}).get('allowed') is True)
    return (active.get('action') in {'CHANGE_LANE','OVERTAKE'} and prepared
        and not maneuver_started and origin_side is None)


def require_acquired_vehicle_before_lane_change(contract, steps, active_step_id,
                                                proofs, *, maneuver_started):
    """Require an observed referent before starting a dependent overtaking merge."""
    from copy import deepcopy
    result=deepcopy(contract or {})
    active=next((s for s in steps if s['step_id']==active_step_id),{})
    by_id={s['step_id']:s for s in steps}
    required=[s for s in steps if _requires_passing_target(s,active)
        and _depends_on_step(s,active_step_id,by_id)]
    if active.get('action')!='CHANGE_LANE' or maneuver_started or not required:
        return result
    def acquired_target(step):
        proof=proofs.get(step['step_id']) or {}
        observation=proof.get('target_observation') or {}
        acquired=(proof.get('valid') is True and proof.get('target_id') is not None
            and observation.get('measured_in_current_frame') is True)
        if PassingMemory.category(step)=='cyclist':
            road_ahead=observation.get('road_ahead_m')
            road_lateral=observation.get('road_lateral_m')
            if (observation.get('acquisition_geometry_valid') is True
                    and isinstance(road_ahead,(int,float)) and not isinstance(road_ahead,bool)
                    and math.isfinite(road_ahead)
                    and isinstance(road_lateral,(int,float)) and not isinstance(road_lateral,bool)
                    and math.isfinite(road_lateral)):
                ahead,lateral=road_ahead,road_lateral
            else:
                ahead,lateral=observation.get('ahead_m'),observation.get('lateral_m')
            categories=observation.get('observed_categories') or ()
            geometry_valid=(isinstance(ahead,(int,float)) and not isinstance(ahead,bool)
                and math.isfinite(ahead) and 0.0<ahead<=35.0
                and isinstance(lateral,(int,float)) and not isinstance(lateral,bool)
                and math.isfinite(lateral) and abs(lateral)<=4.5)
            acquired=(acquired and geometry_valid and 'cyclist' in categories)
        return acquired

    acquired=all(acquired_target(step) for step in required)
    result['passing_target_acquisition']=dict(acquired=acquired,
        source='same_frame_rgb_lidar_target_before_maneuver',policy_truth_access=False)
    if not acquired:
        result.update(can_execute=False,status='UNKNOWN')
        reasons=list(result.get('reason_codes') or [])
        if 'passing_target_not_acquired' not in reasons:reasons.append('passing_target_not_acquired')
        result['reason_codes']=reasons
    return result


def apply_passing_target_acquisition(risk, steps, active_step_id, proofs, *, maneuver_started):
    """Conjoin referent readiness with every lane-authority consumer atomically."""
    from copy import deepcopy
    result=deepcopy(risk)
    contract=require_acquired_vehicle_before_lane_change(
        result.get('target_lane_contract'),steps,active_step_id,proofs,
        maneuver_started=maneuver_started)
    result['target_lane_contract']=contract
    acquisition=contract.get('passing_target_acquisition') or {}
    if acquisition.get('acquired') is not False:
        return result
    direction=contract.get('direction')
    if direction not in {'left','right'}:
        active=next((step for step in steps if step['step_id']==active_step_id),{})
        direction=str((active.get('parameters') or {}).get('direction','')).lower()
    if direction not in {'left','right'}:
        return result
    judgment=result.setdefault('lane_change',{}).setdefault(direction,{})
    judgment['is_safe']=False
    judgment['reason_codes']=list(dict.fromkeys([
        *(judgment.get('reason_codes') or []),'passing_target_not_acquired']))
    from lightweight_vla_adapter.src.task_feasibility import predicate
    task='CHANGE_LANE_'+direction.upper()
    result.setdefault('task_feasibility',{})[task]=predicate(task,False,known=False,
        reasons=('passing_target_not_acquired',),evidence=deepcopy(contract))
    return result


from scene_understanding.realtime_perception.cyclist_detection import merge_cyclist_expert


def cyclist_expert_view_names(document, available_views, bound, tracks, *, timestamp_s, ego_rear_m):
    """Limit the specialist head to the target's current longitudinal views.

    Unknown identity or geometry deliberately falls back to every available view.
    The regular detector still runs on all views, and this only controls the
    supplemental cyclist classifier.
    """
    views=tuple(view for view in available_views if view in {'front','left','right','rear'})
    if not any(PassingMemory.category(step)=='cyclist'
               for step in (document.get('intent') or {}).get('steps',())):
        return ()
    request_id=document.get('request_id')
    for step in (document.get('intent') or {}).get('steps',()):
        if PassingMemory.category(step)!='cyclist':
            continue
        binding=bound.get((request_id,step.get('step_id')))
        track=(tracks.get(binding.get('target_id')) if binding else None)
        if not isinstance(track,dict):
            continue
        last_seen=track.get('last_seen_s')
        if not isinstance(last_seen,(int,float)) or not 0.<=timestamp_s-last_seen<=.25:
            continue
        geometry=track.get('acquisition_geometry') or track
        ahead=geometry.get('ahead_m')
        aliases=track.get('image_aliases') or {}
        fresh_aliases={view for (view,_),seen in aliases.items()
                       if view in views and isinstance(seen,(int,float))
                       and 0.<=timestamp_s-seen<=.25}
        if (isinstance(ahead,(int,float)) and ahead>ego_rear_m+4.
                and 'front' in fresh_aliases and 'front' in views):
            return ('front',)
        if (isinstance(ahead,(int,float)) and ahead<-(ego_rear_m+4.)
                and 'rear' in fresh_aliases and 'rear' in views):
            return ('rear',)
    return views


class PassingConditionObserver:
    def __init__(self, detector, artifact_dir=None, front_detector=None, cyclist_detector=None,
                 cyclist_completion_mode='measured_rear_clearance'):
        from scene_understanding.realtime_perception.tracker import ByteTrackAdapter
        self.detector=detector
        self.front_detector=front_detector
        self.cyclist_detector=cyclist_detector
        if cyclist_completion_mode not in {'measured_rear_clearance','encounter_window'}:
            raise ValueError('Unknown cyclist completion mode')
        self.cyclist_encounter=None
        if cyclist_completion_mode=='encounter_window':
            from control.cyclist_encounter import CyclistEncounterWindow
            self.cyclist_encounter=CyclistEncounterWindow()
        self.trackers={view:ByteTrackAdapter(frame_rate=20) for view in ('front','left','right','rear')}
        self.memory=PassingMemory();self.last_sensor_frame=None
        self.artifact_dir=artifact_dir;self.saved_facts=set();self.saved_bindings=set()
        self.loss_artifacts=0;self.last_loss_artifact_s=-float('inf')
        self.unbound_artifacts=0
        self.sparse_cyclist_support={}
        batch=getattr(self.detector,'detect_many',None)
        if callable(batch):
            batch([np.zeros((640,640,3),dtype=np.uint8) for _ in range(4)])

    def detect_views(self, usable, shared_front=None, *, cyclist_expert=False,
                     cyclist_expert_views=None):
        outputs=[None]*len(usable)
        regular=[]
        for index,(view,rgb,pose,fov) in enumerate(usable):
            image=np.ascontiguousarray(rgb[:,:,::-1])
            if (view=='front' and shared_front is not None
                    and shared_front.get('source')=='synchronized_normal_front_capture'
                    and shared_front.get('image') is rgb):
                outputs[index]=list(shared_front['detections'])
            elif view=='front' and self.front_detector is not None:
                outputs[index]=self.front_detector.detect(image)[0]
            else:
                regular.append((index,image))
        batch=getattr(self.detector,'detect_many',None)
        if regular and callable(batch):
            detected,_=batch([image for index,image in regular])
        else:
            detected=[self.detector.detect(image)[0] for index,image in regular]
        for (index,image),boxes in zip(regular,detected):outputs[index]=boxes
        expert_views=(set(cyclist_expert_views) if cyclist_expert_views is not None
                      else {view for view,_,_,_ in usable})
        expert_indices=[index for index,(view,_,_,_) in enumerate(usable)
                        if view in expert_views]
        if expert_indices and cyclist_expert and self.cyclist_detector is not None:
            images=[np.ascontiguousarray(usable[index][1][:,:,::-1]) for index in expert_indices]
            batch=getattr(self.cyclist_detector,'detect_many',None)
            if callable(batch):expert,_=batch(images)
            else:expert=[self.cyclist_detector.detect(image)[0] for image in images]
            for index,extra in zip(expert_indices,expert):
                outputs[index]=merge_cyclist_expert(outputs[index],extra,authoritative=True)
        return outputs

    def observe(self,document,views,cloud,road,*,frame_id,sensor_frame,observed_at_s,
                timestamp_s,ego_xy,heading,ego_rear_m,corridor_clear,cruise_speed_kmh=30.,passing_origin_side=None,
                target_binding_allowed=True,scan=True,shared_front=None,
                contact_count=None,progress_m=None,ego_speed_kmh=None):
        encounter=getattr(self,'cyclist_encounter',None)
        only_cyclists=all(PassingMemory.category(step)=='cyclist'
            for step in document['intent']['steps'])
        if encounter is not None and only_cyclists and not encounter.needs_scan(document):
            scan=False
        measurements=[];visual_detections=[]
        fresh=(isinstance(observed_at_s,(int,float)) and math.isfinite(observed_at_s)
            and 0<=timestamp_s-observed_at_s<=.25 and cloud is not None)
        if fresh and sensor_frame!=self.last_sensor_frame:
            measurements=[];surface=object_surface_points(cloud,road) if scan else None
            usable=[(view,rgb,pose,fov) for view,(rgb,pose,fov) in views.items()
                    if scan and rgb is not None and pose is not None]
            shared=(shared_front if shared_front is not None
                and shared_front.get('sensor_frame')==sensor_frame else None)
            specialist_views=cyclist_expert_view_names(document,
                (view for view,_,_,_ in usable),self.memory.bound,self.memory.tracks,
                timestamp_s=timestamp_s,ego_rear_m=ego_rear_m)
            outputs=self.detect_views(usable,shared,cyclist_expert=bool(specialist_views),
                cyclist_expert_views=specialist_views)
            visual_detections=[dict(view=view,category=d.category,confidence=d.confidence,
                bbox_xyxy=d.bbox_xyxy) for (view,rgb,pose,fov),detections in zip(usable,outputs)
                for d in detections]
            for (view,rgb,pose,fov),detections in zip(usable,outputs):
                include_riders=any(t['category']=='cyclist' and t.get('motion_history_established')
                    and 0<=observed_at_s-t['last_seen_s']<=8.+1e-8 for t in self.memory.tracks.values())
                selected=semantic_body_nms(detections,include_riders=include_riders)
                tracks=self.trackers[view].update(selected)
                for track in tracks:
                    if track['age_frames']<2:continue
                    box=track['bbox_xyxy'];width,height=rgb.shape[1],rgb.shape[0]
                    cropped=not (2<box[0]<box[2]<width-2 and 2<box[1]<box[3]<height-2)
                    sparse=track['category']=='cyclist'
                    body=lift_box_body(surface,pose,(width,height),fov,box,
                        allow_partial=True,sparse_cyclist=sparse)
                    if body is None:continue
                    support_mode='single_frame'
                    support_frames=1
                    if sparse and len(body)<4:
                        key=(view,track['track_id'])
                        history,confirmed=sparse_body_consensus(
                            self.sparse_cyclist_support.get(key,[]),body,observed_at_s)
                        self.sparse_cyclist_support[key]=history
                        if not confirmed:continue
                        support_mode='temporally_confirmed_sparse'
                        support_frames=len(history)
                    elif sparse:
                        self.sparse_cyclist_support.pop((view,track['track_id']),None)
                    measurements.append(dict(view=view,local_id=track['track_id'],category=track['category'],
                        position=np.median(body,axis=0),surface=body,bbox_xyxy=box,box_cropped=cropped,
                        appearance=body_appearance(rgb,body,pose,fov),
                        body_support_mode=support_mode,body_support_frames=support_frames))
            self.sparse_cyclist_support={key:value for key,value in self.sparse_cyclist_support.items()
                if value and observed_at_s-value[-1][0]<=.2+1e-8}
            self.memory.update(measurements,frame=sensor_frame,timestamp_s=observed_at_s,
                ego_xy=ego_xy,heading=heading,road=road)
            self.last_sensor_frame=sensor_frame
        result=self.memory.proofs(document,frame_id=frame_id,frame=sensor_frame,
            timestamp_s=timestamp_s,observed_at_s=observed_at_s,ego_xy=ego_xy,heading=heading,
            ego_rear_m=ego_rear_m,corridor_clear=corridor_clear and fresh,
            cruise_speed_kmh=cruise_speed_kmh,passing_origin_side=passing_origin_side,
            target_binding_allowed=target_binding_allowed)
        if encounter is not None:
            result.update(encounter.observe(document,self.memory.tracks,
                frame_id=frame_id,sensor_frame=sensor_frame,timestamp_s=timestamp_s,
                observed_at_s=observed_at_s,progress_m=progress_m,ego_speed_kmh=ego_speed_kmh,
                expected_speed_kmh=cruise_speed_kmh,heading=heading,contact_count=contact_count))
        if not fresh:
            for proof in result.values():
                proof.update(valid=False,satisfied=False,can_proceed=False,status='UNKNOWN',
                             reason='passing_sensor_bundle_unavailable')
        if self.artifact_dir:
            from pathlib import Path
            from PIL import Image
            import json
            folder=Path(self.artifact_dir)
            for proof in result.values():
                observation=proof.get('target_observation') or {}
                lost=(proof.get('target_id') and not observation.get('measured_in_current_frame')
                    and observation.get('last_seen_s') is not None
                    and timestamp_s-observation['last_seen_s']>.3)
                unbound=(not proof.get('target_id') and proof.get('condition')=='CYCLIST'
                    and (self.unbound_artifacts<4 or any(d['category']=='cyclist' for d in visual_detections)))
                if (fresh and (lost or unbound)
                        and self.loss_artifacts<20
                        and timestamp_s-self.last_loss_artifact_s>=(5. if unbound else 2.)):
                    folder.mkdir(parents=True,exist_ok=True)
                    stem=f'loss_{sensor_frame}_{self.loss_artifacts}'
                    saved_views=[]
                    for view,(rgb,pose,fov) in views.items():
                        if rgb is None or pose is None:continue
                        Image.fromarray(rgb).save(folder/f'{stem}_{view}.png')
                        saved_views.append(dict(view=view,inverse_camera=np.asarray(pose).tolist(),fov=fov))
                    np.savez_compressed(folder/f'{stem}.npz',world_points=cloud,road=road,
                        ego_xy=ego_xy,heading=heading)
                    diagnostic_measurements=[dict(view=m['view'],local_id=m['local_id'],
                        category=m['category'],position=np.asarray(m['position']).tolist(),
                        bbox_xyxy=m['bbox_xyxy'],box_cropped=m.get('box_cropped',False),
                        body_support_mode=m.get('body_support_mode','dense_current_frame'),
                        body_support_frames=m.get('body_support_frames',1),
                        surface_returns=len(m.get('surface',())),
                        associated_target_id=self.memory.aliases.get((m['view'],m['local_id'])))
                        for m in measurements]
                    (folder/f'{stem}.json').write_text(json.dumps(dict(proof,views=saved_views,
                        diagnostic_state='TARGET_UNBOUND' if unbound else 'TARGET_OCCLUDED',
                        visual_detections=visual_detections,
                        current_body_measurements=diagnostic_measurements),indent=2))
                    self.loss_artifacts+=1;self.last_loss_artifact_s=timestamp_s
                    self.unbound_artifacts+=int(unbound)
                binding_key=(proof['request_id'],proof['step_id'],proof.get('target_id'))
                if (proof.get('target_id') and observation.get('measured_in_current_frame')
                        and binding_key not in self.saved_bindings):
                    track=self.memory.tracks.get(proof['target_id'])
                    folder.mkdir(parents=True,exist_ok=True)
                    stem=f"binding_{sensor_frame}_{len(self.saved_bindings)}"
                    metadata=dict(proof,measurements=[])
                    for index,measurement in enumerate((track or {}).get('measurements',[])):
                        view=measurement['view']
                        if view not in views:continue
                        rgb,pose,fov=views[view]
                        if rgb is None or pose is None:continue
                        Image.fromarray(rgb).save(folder/f'{stem}_{index}.png')
                        np.savez_compressed(folder/f'{stem}_{index}.npz',
                            body_points=measurement['surface'],inverse_camera=pose,
                            ego_xy=ego_xy,heading=heading)
                        metadata['measurements'].append(dict(view=view,
                            bbox_xyxy=measurement['bbox_xyxy'],fov=fov,
                            image_size=[rgb.shape[1],rgb.shape[0]],artifact=f'{stem}_{index}'))
                    (folder/f'{stem}.json').write_text(json.dumps(metadata,indent=2))
                    self.saved_bindings.add(binding_key)
                fact=proof.get('completed_event_fact')
                if not fact or not proof['satisfied']:continue
                key=(fact['source_request_id'],fact['source_step_id'])
                if key in self.saved_facts:continue
                track=self.memory.tracks.get(fact['target_id'])
                rear=[m for m in (track or {}).get('measurements',[]) if m['view']=='rear' and not m.get('box_cropped',False)]
                if not rear or 'rear' not in views:continue
                rgb,pose,fov=views['rear'];folder.mkdir(parents=True,exist_ok=True)
                stem=f"{sensor_frame}_{len(self.saved_facts)}"
                Image.fromarray(rgb).save(folder/f'{stem}.png')
                np.savez_compressed(folder/f'{stem}.npz',body_points=np.concatenate([m['surface'] for m in rear]),
                    inverse_camera=pose,ego_xy=ego_xy,heading=heading)
                metadata=dict(proof,boxes=[m['bbox_xyxy'] for m in rear],fov=fov,
                    image_size=[rgb.shape[1],rgb.shape[0]],ego_rear_m=ego_rear_m)
                (folder/f'{stem}.json').write_text(json.dumps(metadata,indent=2))
                fact['evidence_artifact_stem']=str(folder/stem)
                self.saved_facts.add(key)
        return result
