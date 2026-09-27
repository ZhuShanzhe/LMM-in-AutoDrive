"""Independent episode journal and task assessment; never supplies policy inputs."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

from .catalog import ConfigError, load_catalog, load_episode_catalog
from .monitor import TaskMonitor
from .safety_events import SafetyLedger
from .task_oracle import load_profile
from .role_journal import RoleJournal, configured_role_anchors
from .traffic_observation import observe_traffic, TrafficDensitySummary
from .truth_capture import ActorBinding, RouteProjector, SnapshotTruthCollector, prepare_lane_fixture


def attach_episode(scene,route,world,ego,output,source_config,initial_route_s_m=0,task_selector='all',run_metadata=None):
    from .turn_fixture import waypoint_route
    if not route or source_config is None:
        raise ConfigError('formal assessment requires an explicit source config and route')
    if not isinstance(route[0],dict):
        route=waypoint_route([item[0] for item in route])
    settings=world.get_settings()
    metadata=dict(runner=dict(run_metadata or {}),world_settings={
        key:getattr(settings,key,None) for key in ('synchronous_mode','fixed_delta_seconds',
        'substepping','max_substep_delta_time','max_substeps','no_rendering_mode')})
    observer=EpisodeAssessment(scene,route,world.get_map(),ego,output,source_config,initial_route_s_m,task_selector,metadata)
    observer.actor_provider=world.get_actors
    observer.attach_sensors(world)
    return observer


def assessment_outcome(summary):
    """Process result for recorded checks, never a full benchmark certificate."""
    statuses=[v.get('status') for v in summary.get('tasks',{}).values()]
    safety=summary.get('episode_safety',{}).get('status')
    if summary.get('status')=='ASSESSMENT_ERROR':
        code,status=4,'ASSESSMENT_ERROR'
    elif safety=='FAILURE' or any(s in {'FAILURE','TIMEOUT'} for s in statuses):
        code,status=2,'CHECK_FAILED'
    elif (not statuses or safety!='NO_RECORDED_VIOLATION' or
          not any(s=='SUCCESS' for s in statuses) or
          any(s not in {'SUCCESS','NOT_RUN'} for s in statuses)):
        code,status=3,'INCOMPLETE_EVIDENCE'
    else:
        code,status=0,'RECORDED_CHECKS_PASSED'
    return dict(exit_code=code,status=status,scope='recorded_task_and_safety_checks_only',
                full_benchmark_acceptance=False)


def finish_episode(observer):
    """Preserve teardown even if assessment fails; never turn that failure into success."""
    if observer is None:
        return 0
    if hasattr(observer,'final_exit_code'):
        return observer.final_exit_code
    try:
        payload=observer.close()
    except Exception as error:
        try:
            observer.stream.close()
        except Exception:
            pass
        payload=dict(status='ASSESSMENT_ERROR',reason=f'{type(error).__name__}: {error}',benchmark_ready=False)
        print('Independent assessment failed:',payload['reason'])
    outcome=assessment_outcome(payload)
    try:
        if payload.get('status')=='ASSESSMENT_ERROR':
            (observer.output/'summary.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
        (observer.output/'run_outcome.json').write_text(json.dumps(outcome,indent=2),encoding='utf-8')
    except OSError as error:
        print('Independent assessment result could not be saved:',error)
        outcome['exit_code']=4
    observer.final_exit_code=outcome['exit_code']
    return observer.final_exit_code


def pack_actor(actor):
    pose,velocity=actor.get_transform(),actor.get_velocity()
    return dict(x=pose.location.x,y=pose.location.y,z=pose.location.z,yaw=pose.rotation.yaw,
                vx=velocity.x,vy=velocity.y,vz=velocity.z)


def unpack_snapshot(row, location_factory=NS):
    actors={}
    for identity,state in row['actors'].items():
        pose=NS(location=location_factory(**{k:state[k] for k in ('x','y','z')}),rotation=NS(yaw=state['yaw']))
        velocity=NS(x=state['vx'],y=state['vy'],z=state['vz'])
        actors[int(identity)]=NS(get_transform=lambda p=pose:p,get_velocity=lambda v=velocity:v)
    return NS(frame=row['frame'],timestamp=NS(elapsed_seconds=row['sim_time_s']),find=actors.get)


class EpisodeAssessment:
    def __init__(self,scene,route,world_map,ego,output,source_config,initial_route_s_m=0,task_selector='all',run_metadata=None):
        self.catalog=load_episode_catalog(scene, source_config, world_map)
        from .selection import assessment_selection
        self.selected_tasks,self.assessment_tasks=assessment_selection(self.catalog,task_selector)
        if hashlib.sha256(Path(source_config).read_bytes()).hexdigest()!=self.catalog.source_sha256:
            raise ConfigError('assessment source configuration differs from registered catalog')
        if world_map.name.split('/')[-1]!=self.catalog.map_name:
            raise ConfigError('assessment map mismatch')
        self.route=route
        self.projector=RouteProjector(route)
        self.map=world_map
        self.ego=ego
        self.output=Path(output)
        self.output.mkdir(parents=True,exist_ok=False)
        if not route[0]['distance_m']<=initial_route_s_m<=route[-1]['distance_m']:
            raise ConfigError('episode start outside route')
        self.binding=ActorBinding.from_actor(ego,initial_route_s_m)
        self.ledger=SafetyLedger()
        self.sensors=[]
        self.closed=False
        self.frames=0
        self.last_frame=None
        self.location_factory=NS
        self.hint=initial_route_s_m
        self.monitors={}
        self.unavailable={}
        self.profiles={}
        for task in self.assessment_tasks:
            profile=load_profile(self.catalog,task)
            if task.activate_m<initial_route_s_m:
                self.unavailable[task.task_id]='before_segment_start'
            elif profile is None:
                self.unavailable[task.task_id]='missing_or_unbound_profile'
            else:
                self.profiles[task.task_id]=profile
        anchors=configured_role_anchors(self.catalog.events)
        self.role_journal=RoleJournal(self.profiles.values(),anchors)
        self.actor_provider=None
        self.traffic_density=TrafficDensitySummary()
        self.traffic_segments={}
        map_xml=world_map.to_opendrive() if hasattr(world_map,'to_opendrive') else None
        map_hash=hashlib.sha256(map_xml.encode('utf-8')).hexdigest() if map_xml else None
        if map_xml:
            (self.output/'map.xodr').write_bytes(map_xml.encode('utf-8'))
        self.stream=(self.output/'episode_truth.jsonl').open('x',encoding='utf-8')
        (self.output/'route.json').write_text(json.dumps(route,indent=2),encoding='utf-8')
        self.manifest=dict(scene_id=scene,source_sha256=self.catalog.source_sha256,
            geometry_binding=self.catalog.geometry_binding,
            route_sha256=hashlib.sha256(json.dumps(route,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest(),
            run_metadata=json.loads(json.dumps(run_metadata or {},allow_nan=False)),
            assessment_profile_sha256={identity:hashlib.sha256(
                json.dumps(profile,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()
                for identity,profile in self.profiles.items()},
            reproducibility_scope='recorded inputs and settings; not deterministic trajectory certification',
            task_selector=task_selector,selected_task_ids=[task.task_id for task in self.selected_tasks],
            evidence_task_ids=[task.task_id for task in self.assessment_tasks],
            map_name=world_map.name,map_geometry_sha256=map_hash,ego_binding=asdict(self.binding),
            mode='formal_runner_observer',policy_inputs=False,
            required_roles=self.role_journal.expected,
            safety_scope=['collision','solid_lane_marking'])
        (self.output/'manifest.json').write_text(json.dumps(self.manifest,indent=2),encoding='utf-8')

    def attach_sensors(self,world):
        import carla
        try:
            for kind in ('collision','lane_invasion'):
                sensor=world.spawn_actor(world.get_blueprint_library().find('sensor.other.'+kind),
                                         carla.Transform(),attach_to=self.ego)
                self.sensors.append(sensor)
                if kind=='collision':
                    sensor.listen(lambda e:self.ledger.record('collision',int(e.frame),other_actor_id=e.other_actor.id))
                else:
                    sensor.listen(lambda e:self.ledger.record('lane_invasion',int(e.frame),
                        illegal=any('Solid' in str(m.type) or str(m.type)=='Curb' for m in e.crossed_lane_markings)))
        except Exception:
            for sensor in self.sensors:
                if sensor.is_listening:
                    sensor.stop()
                sensor.destroy()
            self.sensors.clear()
            self.stream.close()
            raise

    def observe(self,snapshot):
        if self.closed:
            raise RuntimeError('episode assessment closed')
        if self.last_frame is not None and snapshot.frame<=self.last_frame:
            raise ConfigError('duplicate or nonmonotonic episode frame')
        actor=snapshot.find(self.binding.actor_id)
        if actor is None:
            raise ConfigError('episode ego missing')
        point=actor.get_transform().location
        self.location_factory=type(point)
        self.hint=self.projector.project(point,self.hint)['route_s_m']
        row=dict(frame=snapshot.frame,sim_time_s=snapshot.timestamp.elapsed_seconds,
                 route_s_m=self.hint,actors={str(self.binding.actor_id):pack_actor(actor)})
        live_actors=list(self.actor_provider()) if self.actor_provider is not None else []
        if self.actor_provider is not None:
            vehicle_ids=[a.id for a in live_actors if getattr(a,'type_id','').startswith('vehicle.')]
            observation=observe_traffic(snapshot,self.binding.actor_id,vehicle_ids,self.map)
            observation['population']='all_world_vehicles_including_task_actors'
            row['traffic_observation']=observation
            self.traffic_density.update(observation)
            segment=int(self.hint//500)*500
            self.traffic_segments.setdefault(segment,TrafficDensitySummary()).update(observation)
        if self.role_journal.expected:
            roles,actors=self.role_journal.capture(snapshot,live_actors,self.projector,
                                                  self.hint,pack_actor)
            if str(self.binding.actor_id) in actors:
                raise ConfigError('task role must not bind ego')
            row['roles']=roles
            row['actors'].update(actors)
        self.stream.write(json.dumps(row,allow_nan=False)+'\n')
        self.stream.flush()
        self.frames+=1
        self.last_frame=snapshot.frame

    def _prepare(self,task_id,profile,snapshot,progress,role_states=None):
        fixture={'steps':{}}
        try:
            for dependency in profile.get('requires_task_success',[]):
                prior=self.monitors.get(dependency)
                result=prior.oracle.result() if prior is not None else {}
                completion=next((e for e in result.get('evidence',[]) if e['event']=='SUCCESS'),None)
                if result.get('status')!='SUCCESS' or completion is None or completion['frame']>=snapshot.frame:
                    raise ConfigError('prerequisite not independently completed: '+dependency)
                fixture.setdefault('prerequisites',{})[dependency]=dict(status='SUCCESS',
                    source='independent_task_oracle',source_sha256=self.catalog.source_sha256,
                    completion_frame=completion['frame'])
            roles={}
            required_roles = {step['target_role'] for step in profile['steps'] if 'target_role' in step}
            required_roles.update(role for step in profile['steps'] for role in step.get('target_roles', []))
            for role in sorted(required_roles):
                state=(role_states or {}).get(role,{})
                if state.get('status')!='BOUND':
                    raise ConfigError(f'role {role} not valid at task entry: '+state.get('reason','not recorded'))
                roles[role]=ActorBinding(**state['binding'])
                if 'route_s_m' in state:
                    roles[role]=ActorBinding(**dict(state['binding'],initial_route_s_m=state['route_s_m']))
                if snapshot.find(roles[role].actor_id) is None:
                    raise ConfigError(f'role {role} absent at task entry')
            from .turn_fixture import bind_junction_sequence
            later=[task.activate_m for task in self.catalog.tasks if task.activate_m>profile['activate_m']]
            junction_end=profile.get('end_route_s_m',min(later) if later else self.route[-1]['distance_m'])
            fixture['steps'].update(bind_junction_sequence(self.route,self.map,self.location_factory,
                profile['steps'],progress,junction_end))
            for i,step in enumerate(profile['steps']):
                if step['kind']=='maintain_interval' and step.get('keep_lane'):
                    from .lane_continuity import build_lane_corridor
                    fixture['steps'][str(i)]=dict(lane_corridor=build_lane_corridor(
                        self.map,self.route,step['from_route_s_m'],self.location_factory,
                        step['until_route_s_m']))
                elif step['kind']=='speed' and step.get('keep_lane'):
                    from .lane_continuity import trace_lane_corridor
                    from .truth_capture import lane_key
                    end=profile.get('end_route_s_m',self.route[-1]['distance_m'])
                    evidence=trace_lane_corridor(self.map,self.route,profile['activate_m'],
                                                self.location_factory,end)
                    segments=evidence['lane_corridor']
                    current=self.map.get_waypoint(snapshot.find(self.binding.actor_id).get_transform().location)
                    allowed={key for s in segments if s['start_m']<=progress<=s['end_m'] for key in s['lane_keys']}
                    if current is None or lane_key(current) not in allowed:
                        raise ConfigError('current lane differs from speed-task route corridor')
                    fixture['steps'][str(i)]=evidence
                elif step['kind']=='destination':
                    end=self.route[-1]
                    if end['distance_m']<self.catalog.route_length_m:
                        raise ConfigError('recorded route ends before required scene distance')
                    fixture['steps'][str(i)]=dict(route_end_s_m=end['distance_m'],
                        position_m={key:end[key] for key in ('x','y','z')})
                    if step.get('keep_lane'):
                        from .lane_continuity import build_lane_corridor
                        fixture['steps'][str(i)]['lane_corridor']=build_lane_corridor(
                            self.map,self.route,profile['activate_m'],self.location_factory)
                        current=self.map.get_waypoint(snapshot.find(self.binding.actor_id).get_transform().location)
                        segments=fixture['steps'][str(i)]['lane_corridor']
                        allowed={key for segment in segments if segment['start_m']<=progress<=segment['end_m']
                                 for key in segment['lane_keys']}
                        from .truth_capture import lane_key
                        if current is None or lane_key(current) not in allowed:
                            raise ConfigError('current lane differs from destination route corridor')
                elif step['kind'] in {'lane_change','guarded_lane_change'}:
                    location=snapshot.find(self.binding.actor_id).get_transform().location
                    if i>0:
                        previous = fixture['steps'].get(str(i-1), {})
                        if 'start_route_s_m' in step:
                            anchor=step['start_route_s_m']
                        elif profile['steps'][i-1]['kind'] in {'turn','straight_junction'} and 'junction_end_m' in previous:
                            anchor=previous['junction_end_m']+5.0
                        elif all(s['kind'] in {'speed','speed_ceiling','speed_change','lane_hold','wait_clear'} for s in profile['steps'][:i]):
                            anchor=progress
                        else:
                            raise ConfigError('sequential lane change requires explicit route entry')
                        if not progress<=anchor<=self.route[-1]['distance_m']:
                            raise ConfigError('sequential lane entry outside remaining route')
                        point=min(self.route,key=lambda p:abs(p['distance_m']-anchor))
                        location=self.location_factory(**{k:point[k] for k in ('x','y','z')})
                    fixture['steps'][str(i)]=prepare_lane_fixture(self.map,location,step['direction'])
                elif step['kind'] in {'yield_pedestrian','wait_clear'}:
                    from .event_fixture import crosswalk_fixture, route_crossing_fixture
                    target_roles=set(step.get('target_roles',[step.get('target_role')]))
                    events=[e for e in self.catalog.events
                            if target_roles <= set(e.get('ground_truth',{}).get('actor_roles',[]))
                            and ('crosswalk_polygon_index' in e or e.get('scenario')=='temporary_worker_crossing'
                                 or e.get('kind')=='bus_stop')]
                    if len(events)!=1:
                        raise ConfigError('pedestrian requires one source-configured crossing event')
                    event=events[0]
                    if event.get('kind')=='bus_stop':
                        from .event_fixture import bus_passenger_fixture
                        fixture['steps'][str(i)]=bus_passenger_fixture(
                            self.map,self.route,event,target_roles,self.location_factory)
                        continue
                    if event.get('scenario')=='temporary_worker_crossing':
                        workers=[w for w in event.get('workers',[]) if w.get('role_name') in target_roles]
                        anchors={w['start_s_m'] for w in workers}
                        if len(workers)!=len(target_roles) or len(anchors)!=1:
                            raise ConfigError('crossing roles require one configured unmarked crossing anchor')
                        fixture['steps'][str(i)]=route_crossing_fixture(
                            self.map,self.route,anchors.pop(),self.location_factory)
                        continue
                    if 'crosswalk_polygon_index' not in event or 'anchor_progress_m' not in event:
                        raise ConfigError('crossing event lacks explicit geometry')
                    fixture['steps'][str(i)]=crosswalk_fixture(self.map,self.route,
                        event['crosswalk_polygon_index'],event['anchor_progress_m'])
            for constraint in profile.get('constraints',[]):
                if constraint.get('keep_lane'):
                    from .lane_continuity import build_lane_corridor
                    fixture.setdefault('constraints',{})[constraint['id']]=dict(
                        lane_corridor=build_lane_corridor(self.map,self.route,
                            constraint['from_route_s_m'],self.location_factory,constraint['until_route_s_m']))
            binding=ActorBinding(**dict(asdict(self.binding),initial_route_s_m=progress))
            collector=SnapshotTruthCollector(profile,self.route,self.map,binding,roles,fixture,'formal_route')
            self.monitors[task_id]=TaskMonitor(collector,self.output/'tasks'/task_id)
        except (ConfigError, KeyError, TypeError, ValueError) as error:
            self.unavailable[task_id]='invalid_entry_fixture: '+str(error)

    def close(self):
        if self.closed:
            return self.summary
        self.stream.close()
        cleanup_errors=[]
        for sensor in self.sensors:
            try:
                if sensor.is_listening:
                    sensor.stop()
            except RuntimeError as error:
                cleanup_errors.append(str(error))
            try:
                sensor.destroy()
            except RuntimeError as error:
                cleanup_errors.append(str(error))
        self.sensors.clear()
        if cleanup_errors:
            raise ConfigError('assessment sensor cleanup failed: '+'; '.join(cleanup_errors))
        self.ledger.seal_after_quiet()
        try:
            with (self.output/'episode_truth.jsonl').open(encoding='utf-8') as stream:
                for line in stream:
                    row=json.loads(line)
                    snapshot=unpack_snapshot(row,self.location_factory)
                    for identity,profile in self.profiles.items():
                        if identity in self.unavailable or row['route_s_m']<profile['activate_m']:
                            continue
                        if identity not in self.monitors:
                            self._prepare(identity,profile,snapshot,row['route_s_m'],row.get('roles'))
                        if identity in self.monitors:
                            needed=self.monitors[identity].collector.bindings.keys()-{'ego'}
                            invalid=[role for role in needed if row.get('roles',{}).get(role,{}).get('status')!='BOUND']
                            self.monitors[identity].observe(snapshot,self.ledger.packet(snapshot.frame),
                                dict(frame=snapshot.frame,complete=True,valid=not invalid,
                                     reason='invalid task roles: '+', '.join(sorted(invalid))))
        finally:
            for monitor in self.monitors.values():
                monitor.close()
        results={}
        for task in self.assessment_tasks:
            identity=task.task_id
            if identity in self.monitors:
                results[identity]=self.monitors[identity].feedback()
            elif identity in self.unavailable:
                reason=self.unavailable[identity]
                results[identity]=dict(status='NOT_RUN' if reason=='before_segment_start' else
                                       'SCENE_INVALID' if reason.startswith('invalid_entry_fixture:')
                                       else 'UNSUPPORTED',reason=reason)
            else:
                results[identity]=dict(status='NOT_REACHED',reason='activation_not_observed')
        selected_ids={task.task_id for task in self.selected_tasks}
        prerequisite_results={key:value for key,value in results.items() if key not in selected_ids}
        results={key:value for key,value in results.items() if key in selected_ids}
        successful=sum(v['status']=='SUCCESS' for v in results.values())
        verified=sum(v.get('instruction_status')=='SUCCESS' for v in results.values())
        self.summary=dict(**self.manifest,captured_frames=self.frames,final_route_s_m=self.hint,
            tasks=results,prerequisite_tasks=prerequisite_results,success_count=successful,total_tasks=len(results),
            recorded_criteria_pass_rate=successful/len(results),
            verified_instruction_success_count=verified,
            verified_completion_rate_lower_bound=verified/len(results),
            success_count_scope='registered_criteria_not_full_instruction_semantics',
            assessed_tasks=len(self.monitors),benchmark_ready=False,
            traffic_density=dict(status='RECORDED' if self.traffic_density.frames else 'UNAVAILABLE',
                overall=self.traffic_density.result(),
                route_segments=[dict(from_m=start,until_m=start+500,**summary.result())
                    for start,summary in sorted(self.traffic_segments.items())],
                population='all_world_vehicles_including_task_actors',
                sampling='each observed frame; frame weighted, not distance weighted'),
            role_capture_errors=self.role_journal.errors,
            episode_safety=self.ledger.episode_result(),safety=self.ledger.export(),
            scope='partial_formal_runner_assessment_not_full_acceptance')
        (self.output/'summary.json').write_text(json.dumps(self.summary,ensure_ascii=False,indent=2),encoding='utf-8')
        self.closed=True
        return self.summary
