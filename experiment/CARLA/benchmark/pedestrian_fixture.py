"""Derived unmarked-crossing fixture for the scene-three yield assessment."""
import math

from .catalog import ConfigError
from .event_fixture import bind_task_roles, route_crossing_fixture
from .speed_fixture import slice_speed_fixture
from .truth_capture import polygons_intersect


class PedestrianFixture:
    def __init__(self, world, route, spec, events):
        import carla
        self.world=world
        self.actors=[]
        self.events=[]
        self.ped_started=False
        self.lane_issued=False
        self.clear_since=None
        self.seen_conflict=False
        self.role=spec['steps'][0]['target_role']
        candidates=[(event,worker) for event in events if event.get('scenario')=='temporary_worker_crossing'
                    for worker in event.get('workers',[]) if worker.get('role_name')==self.role]
        if len(candidates)!=1:
            raise ConfigError('yield fixture requires a unique configured crossing worker')
        event,worker=candidates[0]
        self.source_event=event
        self.worker_config=worker
        self.route,self.spec,self.offset=slice_speed_fixture(route,spec,world.get_map(),carla.Location,pre_roll_m=30)
        self.crossing=route_crossing_fixture(world.get_map(),self.route,worker['start_s_m']-self.offset,carla.Location)
        self.fixture={'steps':{'0':self.crossing}}
        self.speed=float(event['crossing_behavior']['speed_mps'])
        if not math.isfinite(self.speed) or self.speed<=0:
            raise ConfigError('invalid worker speed')
        self.bindings={}

    def spawn(self, traffic_manager):
        import carla
        from scenarios.complex.town05_scene2 import ScriptedWalker
        self.tm=traffic_manager
        point=min(self.route,key=lambda p:abs(p['distance_m']-self.crossing['route_anchor_m']))
        from scene3_town05_route import Town05RouteMapAdapter
        waypoint=self.world.get_map().get_waypoint(carla.Location(x=point['x'],y=point['y'],z=point['z']))
        adapter=Town05RouteMapAdapter(self.world.get_map(),[(waypoint,None)],[self.worker_config['start_s_m']])
        source=adapter.logical_waypoint(self.worker_config['start_lane_id'],self.worker_config['start_s_m'])
        destination=adapter.logical_waypoint(self.worker_config['destination_lane_id'],self.worker_config['start_s_m'])
        if source is None or destination is None:
            raise ConfigError('configured worker start/destination lane unavailable')
        start=carla.Location(x=source.transform.location.x,y=source.transform.location.y,z=source.transform.location.z+.5)
        end=destination.transform.location
        bp=self.world.get_blueprint_library().find('walker.pedestrian.0001')
        bp.set_attribute('role_name',self.role)
        if bp.has_attribute('is_invincible'):
            bp.set_attribute('is_invincible','false')
        actor=self.world.try_spawn_actor(bp,carla.Transform(start))
        if actor is None:
            raise ConfigError('crossing worker spawn occupied')
        self.actors.append(actor)
        self.walker=ScriptedWalker(actor,end,self.speed)
        self.walker.completion_distance_m=.3
        self.bindings=bind_task_roles(self.spec,self.actors,{self.role:self.crossing['route_anchor_m']})

    def tick(self, snapshot, ego, progress, traffic):
        now=snapshot.timestamp.elapsed_seconds
        if not self.ped_started and progress>=self.spec['activate_m']:
            from worker_clearance import worker_trigger_ready
            if worker_trigger_ready(self.crossing['route_anchor_m'],progress,self.source_event):
                self.walker.start()
                self.ped_started=True
                self.events.append(dict(frame=snapshot.frame,event='worker_started',route_s_m=progress))
        self.walker.update()
        actor=snapshot.find(self.walker.actor.id)
        if actor is None:
            raise ConfigError('crossing worker missing')
        conflict=polygons_intersect(self.bindings[self.role].footprint(actor.get_transform()),self.crossing['conflict_polygon_xy'])
        self.seen_conflict |= conflict
        if conflict or not self.seen_conflict:
            self.clear_since=None
        elif self.clear_since is None:
            self.clear_since=now
            self.events.append(dict(frame=snapshot.frame,event='worker_geometrically_clear'))
        clear=self.clear_since is not None and now-self.clear_since>=self.spec['steps'][0]['clear_hold_s']
        self.tm.set_desired_speed(ego,0 if self.ped_started and not clear else 30)
        return dict(pedestrian_conflict=conflict,pedestrian_clear=clear,worker_started=self.ped_started)

    def destroy(self):
        errors=[]
        for actor in self.actors:
            try:
                if actor.is_alive and actor.destroy() is False:
                    errors.append('actor destroy returned False')
            except RuntimeError as error:
                errors.append(str(error))
        self.actors.clear()
        if errors:
            raise RuntimeError('; '.join(errors))
