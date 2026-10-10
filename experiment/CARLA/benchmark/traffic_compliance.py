"""Independent simulator compliance evidence, never a policy input."""
import math


def crossing_evidence(previous, current, line, previous_state, current_state):
    """Front-bumper crossing of the associated lane stop plane."""
    if previous is None:return None
    theta=math.radians(line['yaw_deg']);forward=(math.cos(theta),math.sin(theta))
    lateral=(-forward[1],forward[0]);center=line['xy']
    def coordinates(pose):
        dx,dy=pose['front_xy'][0]-center[0],pose['front_xy'][1]-center[1]
        return dx*forward[0]+dy*forward[1],dx*lateral[0]+dy*lateral[1]
    before,side_before=coordinates(previous);after,side_after=coordinates(current)
    heading=(current['yaw_deg']-line['yaw_deg']+180.)%360.-180.
    if abs(heading)>60. or before>0. or after<=0.:return None
    fraction=-before/(after-before)
    side=side_before+fraction*(side_after-side_before)
    if abs(side)>line['lane_width_m']/2.+.05:return None
    if line['lane_key'] not in {tuple(previous['lane_key']),tuple(current['lane_key'])}:return None
    if math.dist(previous['front_xy'],current['front_xy'])>5.:
        return dict(status='INVALID_DISCONTINUITY',line_id=line['id'])
    states=(previous_state,current_state)
    if all(state=='Red' for state in states):status='RED_LIGHT_VIOLATION'
    elif any(state in {'UNAVAILABLE','CONFLICTING_STATES'} for state in states):status='UNAVAILABLE'
    elif 'Red' in states:status='SIGNAL_TRANSITION_CROSSING'
    else:status='PERMITTED_CROSSING'
    return dict(status=status,line_id=line['id'],previous_signal=previous_state,
        current_signal=current_state,lateral_at_crossing_m=side,
        previous_front_distance_m=before,current_front_distance_m=after)


class TrafficComplianceAudit:
    def __init__(self, ego, world_map, binding):
        self.ego=ego;self.map=world_map;self.binding=binding
        self.lines=[];self.capture_errors=[];self.previous=None;self.previous_signals={}
        self.frames=0;self.speed_frames=0;self.speeding_frames=0;self.max_speed_excess=0.
        self.crossings=[];self.incomplete_frames=0

    def initialize(self, world):
        grouped={}
        try:lights=world.get_actors().filter('traffic.traffic_light*')
        except (AttributeError,RuntimeError) as error:
            self.capture_errors.append(dict(error=str(error)))
            return
        for actor in lights:
            try:
                for waypoint in actor.get_stop_waypoints():
                    pose=waypoint.transform
                    key=(waypoint.road_id,waypoint.section_id,waypoint.lane_id,
                        round(pose.location.x,2),round(pose.location.y,2),round(pose.rotation.yaw,2))
                    if key not in grouped:
                        grouped[key]=dict(id='stopline_'+str(len(grouped)),
                            lane_key=key[:3],xy=[pose.location.x,pose.location.y],
                            yaw_deg=pose.rotation.yaw,lane_width_m=waypoint.lane_width,actors=[])
                    grouped[key]['actors'].append(actor)
            except (AttributeError,RuntimeError) as error:
                self.capture_errors.append(dict(actor_id=actor.id,error=str(error)))
        self.lines=list(grouped.values())

    def observe(self, snapshot):
        actor=snapshot.find(self.binding.actor_id)
        pose=actor.get_transform();velocity=actor.get_velocity()
        speed=3.6*math.sqrt(velocity.x**2+velocity.y**2+velocity.z**2)
        self.frames+=1
        try:limit=float(self.ego.get_speed_limit())
        except (AttributeError,RuntimeError,ValueError):limit=None
        valid_limit=isinstance(limit,float) and math.isfinite(limit) and limit>0
        if valid_limit:
            self.speed_frames+=1;excess=max(0.,speed-limit)
            self.max_speed_excess=max(self.max_speed_excess,excess)
            self.speeding_frames+=int(excess>1e-4)
        else:self.incomplete_frames+=1
        yaw=math.radians(pose.rotation.yaw)
        front=self.binding.half_length_m+self.binding.offset_x_m
        offset=self.binding.offset_y_m
        xy=[pose.location.x+math.cos(yaw)*front-math.sin(yaw)*offset,
            pose.location.y+math.sin(yaw)*front+math.cos(yaw)*offset]
        waypoint=self.map.get_waypoint(pose.location)
        current=dict(front_xy=xy,yaw_deg=pose.rotation.yaw,
            lane_key=(waypoint.road_id,waypoint.section_id,waypoint.lane_id) if waypoint else (),
            frame=snapshot.frame)
        signals={};crossings=[]
        for line in self.lines:
            # A stop plane beyond 12 m cannot be crossed in one ordinary frame.
            nearby=min(math.dist(xy,line['xy']),
                math.dist(self.previous['front_xy'],line['xy']) if self.previous else float('inf'))<=12.
            if not nearby:continue
            states=[]
            for light in line['actors']:
                try:states.append(str(light.get_state()).rsplit('.',1)[-1])
                except RuntimeError:states.append('UNAVAILABLE')
            state=states[0] if states and len(set(states))==1 else 'CONFLICTING_STATES'
            signals[line['id']]=state
            evidence=crossing_evidence(self.previous,current,line,
                self.previous_signals.get(line['id'],'UNAVAILABLE'),state)
            if evidence:
                evidence.update(frame=snapshot.frame,sim_time_s=snapshot.timestamp.elapsed_seconds,
                    traffic_light_actor_ids=[light.id for light in line['actors']])
                crossings.append(evidence);self.crossings.append(evidence)
        self.previous=current;self.previous_signals=signals
        return dict(schema_version='traffic_compliance/1.0',policy_inputs=False,
            speed_kmh=speed,vehicle_api_speed_limit_kmh=limit if valid_limit else None,
            speed_excess_kmh=max(0.,speed-limit) if valid_limit else None,
            nearby_stop_signals=signals,crossings=crossings)

    def result(self):
        violations=sum(row['status']=='RED_LIGHT_VIOLATION' for row in self.crossings)
        uncertain=[row for row in self.crossings if row['status'] not in
            {'RED_LIGHT_VIOLATION','PERMITTED_CROSSING'}]
        complete=(self.frames>0 and self.speed_frames==self.frames and bool(self.lines)
            and not self.capture_errors and not uncertain)
        return dict(scope='Independent snapshot speed and front-bumper stop-line crossing audit; not policy input',
            status='FAILURE' if violations or self.speeding_frames else 'NO_RECORDED_VIOLATION' if complete else 'INCOMPLETE_EVIDENCE',
            frames=self.frames,speed_limit_valid_frames=self.speed_frames,
            speeding_frames=self.speeding_frames,max_speed_excess_kmh=self.max_speed_excess,
            red_light_violations=violations,stop_line_count=len(self.lines),
            crossings=self.crossings,uncertain_crossings=uncertain,capture_errors=self.capture_errors,
            speed_tolerance_kmh=.0001,
            limitations=['speed limit uses simulator vehicle API, not independently reconstructed sign history',
                'stop-line signal transitions remain explicit ambiguous evidence',
                'does not cover every traffic-rule category'])
