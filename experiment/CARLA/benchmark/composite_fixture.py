"""Scene 2 pedestrian/lane/overtake acceptance fixture with explicit baseline control."""
import copy
import math

from .catalog import ConfigError
from .event_fixture import bind_task_roles, crosswalk_candidates, crosswalk_fixture
from .lane_fixture import gap_check
from .truth_capture import prepare_lane_fixture, polygons_intersect, lane_key


class CompositeFixture:
    def __init__(self, world, source_route, source_spec):
        import carla
        self.world=world
        self.actors=[]
        self.events=[]
        self.ped_started=False
        self.lane_issued=False
        self.slow_started=False
        candidates=crosswalk_candidates(world.get_map(),source_route,950.9)
        if not candidates:
            raise ConfigError('no matching official crosswalk for composite fixture')
        candidate=candidates[0]
        origin=min(source_route,key=lambda p:abs(p['distance_m']-(candidate['route_anchor_m']-70)))
        self.offset=origin['distance_m']
        self.route=copy.deepcopy([p for p in source_route if p['distance_m']>=self.offset])
        for point in self.route:
            point['distance_m']-=self.offset
        self.crossing=crosswalk_fixture(world.get_map(),self.route,candidate['polygon_index'],
                                       candidate['route_anchor_m']-self.offset)
        lane=None
        for point in self.route:
            if not self.crossing['route_anchor_m']+20 <= point['distance_m'] <= self.crossing['route_anchor_m']+65:
                continue
            try:
                lane=prepare_lane_fixture(world.get_map(),carla.Location(x=point['x'],y=point['y'],z=point['z']),'LEFT')
            except ConfigError:
                continue
            self.lane_start=point['distance_m']
            break
        if lane is None:
            raise ConfigError('no legal left lane after crossing')
        entries=[]
        targets=[]
        for point in self.route:
            if point['distance_m']<self.lane_start:
                continue
            if point['distance_m']>self.lane_start+150:
                break
            try:
                pair=prepare_lane_fixture(world.get_map(),carla.Location(x=point['x'],y=point['y'],z=point['z']),'LEFT')
            except ConfigError:
                break
            if pair['entry_lane_key'] not in entries:
                entries.append(pair['entry_lane_key'])
            if pair['target_lane_key'] not in targets:
                targets.append(pair['target_lane_key'])
        lane.update(entry_lane_keys=entries,target_lane_keys=targets)
        self.lane=lane
        self.fixture={'steps':{'0':self.crossing,'1':lane}}
        self.spec=copy.deepcopy(source_spec)
        self.spec['activate_m']=10
        self.spec['steps'][1]['start_route_s_m']=self.lane_start
        self.bindings={}
        self.clear_since=None
        self.seen_conflict=False

    def spawn(self, traffic_manager):
        import carla
        from scenarios.complex.town05_scene2 import ScriptedWalker, crosswalk_polygon_endpoints
        self.tm=traffic_manager
        start,end=crosswalk_polygon_endpoints(self.world.get_map(),self.crossing['polygon_index'],inset_m=0)
        length=math.hypot(end.x-start.x,end.y-start.y)
        dx,dy=(end.x-start.x)/length,(end.y-start.y)/length
        start.x-=dx
        start.y-=dy
        end.x+=dx*1.5
        end.y+=dy*1.5
        start.z+=.5
        bp=self.world.get_blueprint_library().find('walker.pedestrian.0001')
        bp.set_attribute('role_name','scene2_crosswalk_pedestrian')
        if bp.has_attribute('is_invincible'):
            bp.set_attribute('is_invincible','false')
        ped=self.world.try_spawn_actor(bp,carla.Transform(start))
        if ped is None:
            raise ConfigError('crosswalk pedestrian spawn occupied')
        self.actors.append(ped)
        self.walker=ScriptedWalker(ped,end,1.45)
        self.walker.completion_distance_m=.3
        point=min(self.route,key=lambda p:abs(p['distance_m']-(self.lane_start+45)))
        bp=self.world.get_blueprint_library().find('vehicle.audi.tt')
        slow_role=next(s['target_role'] for s in self.spec['steps'] if s['kind']=='overtake')
        bp.set_attribute('role_name',slow_role)
        vehicle=self.world.try_spawn_actor(bp,carla.Transform(
            carla.Location(x=point['x'],y=point['y'],z=point['z']+1),carla.Rotation(yaw=point['yaw'])))
        if vehicle is None:
            raise ConfigError('slow vehicle spawn occupied')
        self.actors.append(vehicle)
        vehicle.apply_control(carla.VehicleControl(brake=1,hand_brake=True))
        self.slow=vehicle
        self.bindings=bind_task_roles(self.spec,self.actors,
            {'scene2_crosswalk_pedestrian':self.crossing['route_anchor_m'],
             slow_role:point['distance_m']})

    def tick(self, snapshot, ego, progress, traffic):
        import carla
        now=snapshot.timestamp.elapsed_seconds
        if not self.ped_started and progress>=10:
            self.walker.start()
            self.ped_started=True
            self.events.append(dict(frame=snapshot.frame,event='pedestrian_started'))
        self.walker.update()
        ped=snapshot.find(self.walker.actor.id)
        if ped is None:
            raise ConfigError('pedestrian disappeared')
        conflict=polygons_intersect(self.bindings['scene2_crosswalk_pedestrian'].footprint(ped.get_transform()),
                                    self.crossing['conflict_polygon_xy'])
        self.seen_conflict |= conflict
        if not self.ped_started or not self.seen_conflict or conflict:
            self.clear_since=None
        elif self.clear_since is None:
            self.clear_since=now
            self.events.append(dict(frame=snapshot.frame,event='pedestrian_geometrically_clear'))
        clear=self.clear_since is not None and now-self.clear_since>=.5
        if not clear and progress>=self.crossing['stop_line_route_s_m']-30:
            self.tm.set_desired_speed(ego,0)
        else:
            self.tm.set_desired_speed(ego,30 if not self.lane_issued else 45)
        if clear and not self.slow_started:
            self.slow.apply_control(carla.VehicleControl())
            self.slow.set_autopilot(True,self.tm.get_port())
            self.tm.auto_lane_change(self.slow,False)
            self.tm.set_desired_speed(self.slow,20)
            self.slow_started=True
            self.events.append(dict(frame=snapshot.frame,event='slow_vehicle_released',speed_kmh=20))
        if clear and progress>=self.lane_start and not self.lane_issued:
            decision=gap_check(snapshot,ego,traffic.actors+self.actors,self.world.get_map(),
                               {k:v for k,v in self.lane.items() if k not in {'entry_lane_keys','target_lane_keys'}})
            if decision['clear']:
                path=[]
                for point in self.route:
                    if not progress+5<=point['distance_m']<=progress+150:
                        continue
                    wp=self.world.get_map().get_waypoint(carla.Location(x=point['x'],y=point['y'],z=point['z']))
                    target=wp.get_left_lane()
                    if target is None or lane_key(target) not in self.lane['target_lane_keys']:
                        break
                    path.append(target.transform.location)
                if len(path)<5:
                    raise ConfigError('insufficient post-crossing target lane path')
                self.tm.set_path(ego,path)
                self.tm.force_lane_change(ego,False)
                self.lane_issued=True
                self.events.append(dict(frame=snapshot.frame,event='left_lane_requested'))
        return dict(pedestrian_conflict=conflict,pedestrian_clear=clear,lane_issued=self.lane_issued)

    def destroy(self):
        for actor in self.actors:
            if actor.is_alive:
                actor.destroy()
        self.actors.clear()
