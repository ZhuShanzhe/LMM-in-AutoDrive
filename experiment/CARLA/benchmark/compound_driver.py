"""Explicit simulator-truth baseline for Scene 2 task 3, not a model policy."""
import json
import math

from .catalog import ConfigError
from .event_fixture import crosswalk_fixture
from .lane_fixture import gap_check
from .truth_capture import ActorBinding, lane_key, polygons_intersect, prepare_lane_fixture
from .turn_fixture import waypoint_route


def lane_follow_extension(tail,distance_m=60.0):
    """Follow a unique non-junction successor; never invent a branch choice."""
    extension=[]
    travelled=0.0
    seen={tail.id}
    while travelled<distance_m:
        successors=tail.next(2.0)
        if len(successors)!=1:
            return extension,'ambiguous_or_missing_successor'
        nxt=successors[0]
        if nxt.is_junction or nxt.lane_type!=tail.lane_type or nxt.id in seen:
            return extension,'junction_or_invalid_lane_continuation'
        yaw=(nxt.transform.rotation.yaw-tail.transform.rotation.yaw+180)%360-180
        step=tail.transform.location.distance(nxt.transform.location)
        if not math.isfinite(step) or step<=0 or step>4 or abs(yaw)>30:
            return extension,'discontinuous_lane_continuation'
        extension.append(nxt)
        travelled+=step
        seen.add(nxt.id)
        tail=nxt
    return extension,None


class CompoundDriver:
    def __init__(self, world, ego, agent, route, distances, events, commands, output):
        self.world,self.ego,self.agent=world,ego,agent
        self.route=route
        if len(route)!=len(distances):
            raise ConfigError('baseline route and distance counts differ')
        self.distances=distances
        self.events=events
        self.command=next(c for c in commands if c['id']=='s2_t05_cmd_03')
        if self.command['steps'] != ['YIELD:PEDESTRIAN','ADJUST_SPEED:DECREASE',
                                     'CHANGE_LANE:LEFT_WHEN_SAFE','OVERTAKE:SLOW_VEHICLE']:
            raise ConfigError('compound baseline command contract changed')
        crossing=next(e for e in events.events if e['kind']=='crossing_pedestrian')
        self.crossing=crosswalk_fixture(world.get_map(),waypoint_route([w for w,_ in route]),
            crossing['crosswalk_polygon_index'],crossing['anchor_progress_m'])
        self.pedestrian=events.bindings['scene2_crosswalk_pedestrian']
        self.lead=events.bindings['scene2_compound_slow_vehicle']
        self.ped_binding=ActorBinding.from_actor(self.pedestrian,crossing['anchor_progress_m'])
        self.state='WAITING'
        self.seen_conflict=False
        self.clear_since=None
        self.owns_plan=False
        self.pass_lane_keys=set()
        self.centered_since=None
        self.last_center_sample=None
        self.extension_blocked_tail=None
        self.output=output

    def _extend_pass_plan(self,snapshot):
        plan=list(self.agent.get_local_planner().get_plan())
        if not plan:
            return
        remaining=sum(a[0].transform.location.distance(b[0].transform.location)
                      for a,b in zip(plan,plan[1:]))
        tail=plan[-1][0]
        if remaining>=60 or tail.id==self.extension_blocked_tail:
            return
        extension,reason=lane_follow_extension(tail)
        if extension:
            from agents.navigation.local_planner import RoadOption
            self.agent.set_global_plan([(wp,RoadOption.LANEFOLLOW) for wp in extension],
                                       stop_waypoint_creation=True,clean_queue=False)
            self.pass_lane_keys.update(lane_key(wp) for wp in extension)
        self.extension_blocked_tail=(extension[-1].id if extension else tail.id) if reason else None
        self._record(snapshot,self.state,'passing_horizon_extension',added_points=len(extension),
                     blocked_reason=reason)

    def record_control(self,snapshot,control):
        """Capture the controller target separately from independent task scoring."""
        if not self.owns_plan:
            return
        actor=snapshot.find(self.ego.id)
        if actor is None:
            raise ConfigError('compound control probe has no ego snapshot')
        pose=actor.get_transform()
        planner=self.agent.get_local_planner()
        target=planner.target_waypoint
        current=self.world.get_map().get_waypoint(pose.location)
        def geometry(wp):
            if wp is None:
                return None
            point=wp.transform.location
            yaw=math.radians(wp.transform.rotation.yaw)
            dx=pose.location.x-point.x
            dy=pose.location.y-point.y
            return dict(lane_key=lane_key(wp),x=point.x,y=point.y,z=point.z,
                        yaw_deg=wp.transform.rotation.yaw,
                        lateral_error_m=-dx*math.sin(yaw)+dy*math.cos(yaw),
                        heading_error_deg=(pose.rotation.yaw-wp.transform.rotation.yaw+180)%360-180,
                        distance_m=math.hypot(dx,dy))
        velocity=actor.get_velocity()
        row=dict(frame=snapshot.frame,sim_time_s=snapshot.timestamp.elapsed_seconds,
                 state=self.state,ego_x=pose.location.x,ego_y=pose.location.y,
                 ego_yaw_deg=pose.rotation.yaw,
                 speed_kmh=3.6*math.sqrt(velocity.x**2+velocity.y**2+velocity.z**2),
                 target=geometry(target),current_lane=geometry(current),
                 steer=float(control.steer),throttle=float(control.throttle),brake=float(control.brake),
                 centered_since_s=self.centered_since,simulator_truth_baseline=True)
        with self.output.with_name('compound_control.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')

    def _pass_lane_stable(self,snapshot):
        """Require fresh, sustained lane acquisition before starting a return merge."""
        now=float(snapshot.timestamp.elapsed_seconds)
        actor=snapshot.find(self.ego.id)
        if self.last_center_sample is not None:
            gap=now-self.last_center_sample
            if gap<=0 or gap>0.15:
                self.centered_since=None
        self.last_center_sample=now
        pose=actor.get_transform() if actor is not None else None
        wp=self.world.get_map().get_waypoint(pose.location) if pose is not None else None
        valid=False
        if wp is not None and lane_key(wp) in self.pass_lane_keys and not wp.is_junction:
            yaw=math.radians(wp.transform.rotation.yaw)
            dx=pose.location.x-wp.transform.location.x
            dy=pose.location.y-wp.transform.location.y
            lateral=-dx*math.sin(yaw)+dy*math.cos(yaw)
            heading=(pose.rotation.yaw-wp.transform.rotation.yaw+180)%360-180
            valid=all(math.isfinite(v) for v in (now,lateral,heading)) and abs(lateral)<=0.25 and abs(heading)<=4
        if not valid:
            self.centered_since=None
            return False
        if self.centered_since is None:
            self.centered_since=now
        return now-self.centered_since>=1.5

    def _record(self,snapshot,state,reason,**extra):
        if state!=self.state or extra:
            with self.output.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(frame=snapshot.frame,state=state,reason=reason,
                    sim_time_s=snapshot.timestamp.elapsed_seconds,**extra))+'\n')
        self.state=state

    def update(self,snapshot,progress,default_speed):
        if progress<float(self.command['announce_at_m']) or self.state=='FINISHED':
            return default_speed
        ped=snapshot.find(self.pedestrian.id)
        lead=snapshot.find(self.lead.id)
        ego=snapshot.find(self.ego.id)
        if ped is None or lead is None or ego is None:
            raise ConfigError('compound baseline actor missing')
        if self.state=='MERGE_WAIT':
            self._extend_pass_plan(snapshot)
            if self._pass_lane_stable(snapshot):
                self._resume_route(snapshot,progress)
            return min(default_speed,30)
        conflict=polygons_intersect(self.ped_binding.footprint(ped.get_transform()),
                                   self.crossing['conflict_polygon_xy'])
        now=snapshot.timestamp.elapsed_seconds
        self.seen_conflict |= conflict
        if conflict or not self.seen_conflict:
            self.clear_since=None
        elif self.clear_since is None:
            self.clear_since=now
        clear=self.clear_since is not None and now-self.clear_since>=1.0
        if not self.owns_plan:
            if not clear:
                self._record(snapshot,'YIELD','pedestrian_not_clear')
                return 0.0 if progress>=self.crossing['stop_line_route_s_m']-20 else min(default_speed,30)
            if progress<975.61:
                self._record(snapshot,'APPROACH_LANE','waiting_for_configured_lane_segment')
                return min(default_speed,30)
            world_map=self.world.get_map()
            location=ego.get_transform().location
            fixture=prepare_lane_fixture(world_map,location,'LEFT')
            gap=gap_check(snapshot,self.ego,self.world.get_actors().filter('vehicle.*'),world_map,fixture)
            if not gap['clear']:
                self._record(snapshot,'WAIT_GAP',gap['reason'])
                return min(default_speed,20)
            from agents.navigation.basic_agent import BasicAgent
            plan=BasicAgent._generate_lane_change_path(world_map.get_waypoint(location),
                direction='left',distance_same_lane=5,distance_other_lane=180,lane_change_distance=30)
            if not plan:
                raise ConfigError('official planner could not construct a legal left lane change')
            # Only the post-change lane-follow suffix belongs to the passing lane.
            from agents.navigation.local_planner import RoadOption
            suffix=[]
            for waypoint,option in reversed(plan):
                if option!=RoadOption.LANEFOLLOW:
                    break
                suffix.append(lane_key(waypoint))
            if not suffix:
                raise ConfigError('lane change plan has no passing-lane continuation')
            self.agent.set_global_plan(plan,stop_waypoint_creation=True,clean_queue=True)
            self.pass_lane_keys=set(suffix)
            self.centered_since=None
            self.last_center_sample=None
            self.owns_plan=True
            self._record(snapshot,'LANE_ACQUIRE','official_lane_change_plan',plan_points=len(plan))
        self._extend_pass_plan(snapshot)
        stable=self._pass_lane_stable(snapshot)
        if self.state=='LANE_ACQUIRE':
            if not stable:
                velocity=lead.get_velocity()
                lead_speed=3.6*math.sqrt(velocity.x**2+velocity.y**2+velocity.z**2)
                # Do not initiate longitudinal passing while still acquiring the lane.
                return min(30.0,lead_speed)
            self._record(snapshot,'PASS','passing_lane_stably_acquired')
        pose=ego.get_transform()
        other=lead.get_transform().location
        yaw=math.radians(pose.rotation.yaw)
        ahead=(pose.location.x-other.x)*math.cos(yaw)+(pose.location.y-other.y)*math.sin(yaw)
        if stable and ahead>self.ego.bounding_box.extent.x+self.lead.bounding_box.extent.x+12:
            # Baseline state only. The independent oracle still evaluates actual lane and clearance.
            self._resume_route(snapshot,progress)
        return 45.0

    def _resume_route(self,snapshot,progress):
        """Reconnect only from the route lane or a safe adjacent lane."""
        location=snapshot.find(self.ego.id).get_transform().location
        world_map=self.world.get_map()
        current=world_map.get_waypoint(location)
        index=min(range(len(self.distances)),key=lambda i:abs(self.distances[i]-progress))
        reference=self.route[index][0]
        if current is None:
            raise ConfigError('route continuation has no current waypoint')
        if lane_key(current)!=lane_key(reference):
            direction=None
            for name,neighbor in [('LEFT',current.get_left_lane()),('RIGHT',current.get_right_lane())]:
                if neighbor is not None and lane_key(neighbor)==lane_key(reference):
                    direction=name
                    break
            if direction is None:
                self._record(snapshot,'MERGE_WAIT','route_lane_not_adjacent')
                return False
            try:
                fixture=prepare_lane_fixture(world_map,location,direction)
            except ConfigError:
                self._record(snapshot,'MERGE_WAIT','route_merge_not_legal_here')
                return False
            gap=gap_check(snapshot,self.ego,self.world.get_actors().filter('vehicle.*'),world_map,fixture)
            if not gap['clear']:
                self._record(snapshot,'MERGE_WAIT',gap['reason'])
                return False
        join=next((i for i in range(index,len(self.route)) if self.distances[i]>=progress+60),len(self.route)-1)
        connector=self.agent.trace_route(current,self.route[join][0])
        if not connector or lane_key(connector[-1][0])!=lane_key(self.route[join][0]):
            raise ConfigError('route continuation connector missing or ends on wrong lane')
        plan=list(connector)+list(self.route[join+1:])
        self.agent.set_global_plan(plan,stop_waypoint_creation=True,clean_queue=True)
        self.owns_plan=False
        self._record(snapshot,'FINISHED','route_navigation_restored',join_progress_m=self.distances[join],
                     remaining_plan_points=len(plan))
        return True
