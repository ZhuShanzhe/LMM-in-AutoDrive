"""Derived cyclist acceptance fixture. Scripted baseline, never a model policy."""
import copy
import math

from .catalog import ConfigError
from .event_fixture import bind_task_roles
from .lane_fixture import gap_check
from .overtake_geometry import build_overtake_geometry
from .truth_capture import RouteProjector, lane_key, angle_delta


class OvertakeFixture:
    def __init__(self,world,route,spec,events):
        import carla
        kinds=[step['kind'] for step in spec['steps']]
        if kinds!=['speed_change','lane_change','overtake','return_original_lane']:
            raise ConfigError('unsupported pass-and-return step sequence')
        if spec.get('requires_task_success') or spec.get('constraints'):
            raise ConfigError('cannot omit prerequisite history in isolated overtake')
        self.role=spec['steps'][2]['target_role']
        matches=[e for e in events if self.role in e.get('ground_truth',{}).get('actor_roles',[])
                 and e.get('kind')=='cyclist']
        if len(matches)!=1:
            raise ConfigError('one source cyclist event required')
        self.speed=float(matches[0]['target_speed_kmh'])
        self.world=world
        self.map=world.get_map()
        self.geometry=build_overtake_geometry(self.map,route,carla.Location,
                                             preferred_m=spec['activate_m'],length_m=120)
        self.route=self.geometry['route']
        self.offset=self.geometry['source_entry_m']
        self.spec=copy.deepcopy(spec)
        self.spec['activate_m']=30
        self.spec['end_route_s_m']=self.route[-1]['distance_m']
        pair=dict(self.geometry['outgoing'])
        pair.update(entry_lane_keys=list(dict.fromkeys(p['outgoing']['entry_lane_key'] for p in self.geometry['lane_pairs'])),
                    target_lane_keys=list(dict.fromkeys(p['outgoing']['target_lane_key'] for p in self.geometry['lane_pairs'])))
        self.fixture={'steps':{'1':pair}}
        self.actors=[]
        self.bindings={}
        self.events=[]
        self.stage='PREPARE'
        self.held_since=None
        self.lane_issued=False
        self.projector=RouteProjector(self.route)
        self.target_hint=55
        self.adjustments=dict(local_activate_m=30,cyclist_spawn_local_m=55,
            cyclist_speed_kmh=self.speed,preparation_speed_kmh=30,reduced_speed_kmh=20,
            passing_speed_kmh=35,source_event_position_preserved=False,
            other_event_history_reproduced=False,geometry_only_entry=True)

    def spawn(self,tm):
        import carla
        self.tm=tm
        point=min(self.route,key=lambda p:abs(p['distance_m']-55))
        bp=self.world.get_blueprint_library().find('vehicle.bh.crossbike')
        bp.set_attribute('role_name',self.role)
        actor=self.world.try_spawn_actor(bp,carla.Transform(
            carla.Location(x=point['x'],y=point['y'],z=point['z']+.5),carla.Rotation(yaw=point['yaw'])))
        if actor is None:
            raise ConfigError('cyclist spawn occupied')
        self.actors.append(actor)
        self.target=actor
        actor.apply_control(carla.VehicleControl(brake=1,hand_brake=True))
        self.bindings=bind_task_roles(self.spec,self.actors,{self.role:point['distance_m']})

    def _held(self,condition,now,seconds):
        if not condition:
            self.held_since=None
            return False
        if self.held_since is None:
            self.held_since=now
        return now-self.held_since>=seconds

    def _advance(self,stage,frame):
        self.stage=stage
        self.held_since=None
        self.events.append(dict(frame=frame,event=stage))

    def _path(self,actor,points,progress):
        import carla
        path=[carla.Location(**{k:p[k] for k in ('x','y','z')}) for p in points if p['distance_m']>progress+2]
        if len(path)<2:
            raise ConfigError('insufficient remaining overtake path')
        self.tm.set_path(actor,path)

    def tick(self,snapshot,ego,progress,traffic):
        import carla
        now=snapshot.timestamp.elapsed_seconds
        state=snapshot.find(ego.id)
        target=snapshot.find(self.target.id)
        if state is None or target is None:
            raise ConfigError('overtake actor missing')
        self.target_hint=self.projector.project(target.get_transform().location,self.target_hint)['route_s_m']
        if progress>=self.route[-1]['distance_m']-8 or self.stage=='END_HOLD':
            ego.set_autopilot(False,self.tm.get_port())
            ego.apply_control(carla.VehicleControl(brake=1))
            self.target.set_autopilot(False,self.tm.get_port())
            self.target.apply_control(carla.VehicleControl(brake=1))
            if self.stage!='END_HOLD':
                self._advance('END_HOLD',snapshot.frame)
            return dict(overtake_baseline_stage=self.stage)
        pose=state.get_transform()
        wp=self.map.get_waypoint(pose.location)
        if wp is None:
            raise ConfigError('ego outside mapped overtake corridor')
        v=state.get_velocity()
        speed=3.6*math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z)
        pair=min(self.geometry['lane_pairs'],key=lambda p:abs(p['distance_m']-progress))
        def centered(key):
            center=wp.transform.location
            return lane_key(wp)==key and math.hypot(center.x-pose.location.x,center.y-pose.location.y)<.3 and abs(angle_delta(wp.transform.rotation.yaw,pose.rotation.yaw))<4
        if self.stage=='PREPARE' and progress>=self.spec['activate_m']:
            self.target.apply_control(carla.VehicleControl())
            self.target.set_autopilot(True,self.tm.get_port())
            self.tm.auto_lane_change(self.target,False)
            self.tm.set_desired_speed(self.target,self.speed)
            self._path(self.target,self.route,self.target_hint)
            self.tm.set_desired_speed(ego,20)
            self._advance('DECELERATE',snapshot.frame)
        elif self.stage=='DECELERATE' and self._held(speed<=21,now,.75):
            self._advance('WAIT_LEFT_GAP',snapshot.frame)
        elif self.stage in {'WAIT_LEFT_GAP','WAIT_RETURN_GAP'}:
            outgoing=self.stage=='WAIT_LEFT_GAP'
            fixture=pair['outgoing' if outgoing else 'returning']
            decision=gap_check(snapshot,ego,traffic.actors+self.actors,self.map,fixture)
            if decision['clear']:
                self._path(ego,self.geometry['target_lane_route'] if outgoing else self.route,progress)
                self.tm.force_lane_change(ego,not outgoing)
                self.lane_issued=True
                self._advance('LEFT_SETTLE' if outgoing else 'RETURN_SETTLE',snapshot.frame)
        elif self.stage=='LEFT_SETTLE' and self._held(centered(pair['outgoing']['target_lane_key']),now,1.25):
            self.tm.set_desired_speed(ego,35)
            self._advance('PASS',snapshot.frame)
        elif self.stage=='PASS':
            gap=progress-ego.bounding_box.extent.x-self.target_hint-self.target.bounding_box.extent.x
            if self._held(gap>=18,now,1.25):
                self._advance('WAIT_RETURN_GAP',snapshot.frame)
        elif self.stage=='RETURN_SETTLE' and self._held(centered(pair['outgoing']['entry_lane_key']),now,1.25):
            self._advance('BASELINE_SEQUENCE_FINISHED',snapshot.frame)
        if self.target_hint>=self.route[-1]['distance_m']-8:
            self.target.set_autopilot(False,self.tm.get_port())
            self.target.apply_control(carla.VehicleControl(brake=1))
        return dict(overtake_baseline_stage=self.stage,cyclist_route_s_m=self.target_hint)

    def destroy(self):
        errors=[]
        for actor in self.actors:
            try:
                if actor.is_alive and actor.destroy() is False:
                    raise RuntimeError('actor destroy returned false')
            except Exception as error:
                errors.append(str(error))
        self.actors.clear()
        if errors:
            raise RuntimeError('; '.join(errors))
