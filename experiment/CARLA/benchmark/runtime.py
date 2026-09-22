"""Isolated speed/lane fixtures for simulator-interface acceptance, not model tests."""
import gc
import hashlib
import json
from pathlib import Path
import queue
from types import SimpleNamespace

from .catalog import CONFIG_ROOT, ConfigError, load_catalog
from .task_oracle import load_profile
from .truth_capture import ActorBinding, SnapshotTruthCollector, RouteProjector
from .monitor import TaskMonitor
from .safety_events import SafetyLedger


def run_fixture(scene, selector, route_path, traffic_config_path, output, *,
                      host='localhost',port=2000,seconds=15,seed=42):
    catalog = load_catalog(scene)
    selected = catalog.select(selector)
    if len(selected)!=1:
        raise ConfigError('run supports a single isolated task only')
    spec = load_profile(catalog,selected[0])
    from .capabilities import isolated_capability
    capability=isolated_capability(scene,spec)
    if not capability['adapter_available']:
        raise ConfigError('isolated run unavailable: '+capability['reason'])
    kind=capability['kind']
    source_activate_m = spec['activate_m']
    if not 5 <= seconds <= 120:
        raise ConfigError('fixture duration must be between 5 and 120 seconds')
    if type(seed) is not int or seed < 0:
        raise ConfigError('nonnegative integer seed required')
    route_payload = Path(route_path).read_bytes()
    route = json.loads(route_payload)
    RouteProjector(route)
    if route[0]['distance_m'] != 0:
        raise ConfigError('speed fixture route must start at zero')
    raw = json.loads(Path(traffic_config_path).read_text(encoding='utf-8'))
    map_value=raw.get('map')
    map_name=map_value.get('name') if isinstance(map_value,dict) else map_value
    if map_name is not None and map_name != catalog.map_name:
        raise ConfigError('traffic config map does not match task map')
    traffic_config = dict(raw['traffic'],seed=seed)
    if not traffic_config.get('enabled'):
        raise ConfigError('background traffic required for this fixture')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    (output/'source_spec.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
    from carla_bootstrap import setup_carla_api
    setup_carla_api()
    import carla
    from scenarios.basic.urban_traffic import FixedRouteTraffic

    client = carla.Client(host,port)
    client.set_timeout(60)
    world = client.get_world()
    if world.get_actors().filter('vehicle.*') or world.get_actors().filter('sensor.*') or world.get_actors().filter('walker.*'):
        raise ConfigError('server contains actors; refusing to replace an active world')
    if world.get_map().name.split('/')[-1] != catalog.map_name:
        world = client.load_world(catalog.map_name)
    saved_settings = world.get_settings()
    saved_weather = world.get_weather()
    observation_map = world.get_map()
    ego = traffic = camera = None
    composite = None
    roles = {}
    sensors = []
    snapshot_journal = None
    motion = []
    from .traffic_observation import observe_traffic, TrafficDensitySummary
    traffic_density=TrafficDensitySummary()
    ledger = SafetyLedger()
    fixture = dict(steps={})
    execution = []
    lane_issued = False
    speed_issued = False
    lifecycle = []

    def checkpoint(stage):
        lifecycle.append(stage)
        (output/'lifecycle.json').write_text(json.dumps(lifecycle,indent=2),encoding='utf-8')
    run_info = dict(schema_version='isolated_fixture_run/1.0',task_id=spec['task_id'],
        source_sha256=catalog.source_sha256,route_sha256=hashlib.sha256(route_payload).hexdigest(),
        traffic_config_sha256=hashlib.sha256(Path(traffic_config_path).read_bytes()).hexdigest(),
        seed=seed,controller='traffic_manager_test_baseline',model_inference=False,
        mode='isolated_geometry_fixture_not_original_full_scene',benchmark_ready=False)
    try:
        if kind=='overtake':
            from .overtake_fixture import OvertakeFixture
            composite=OvertakeFixture(world,route,spec,catalog.events)
            route,spec,fixture=composite.route,composite.spec,composite.fixture
            run_info.update(source_activate_m=source_activate_m,source_entry_m=composite.offset,
                local_activate_m=spec['activate_m'],controller='scripted_overtake_acceptance_baseline',
                fixture_adjustments=composite.adjustments,history_equivalent_to_full_run=False)
            (output/'overtake_geometry.json').write_text(json.dumps(composite.geometry,indent=2),encoding='utf-8')
        if kind=='pedestrian':
            from .pedestrian_fixture import PedestrianFixture
            composite=PedestrianFixture(world,route,spec,catalog.events)
            route,spec,fixture=composite.route,composite.spec,composite.fixture
            run_info.update(source_activate_m=source_activate_m,source_entry_m=composite.offset,
                            local_activate_m=spec['activate_m'],controller='scripted_yield_acceptance_baseline',
                            fixture_adjustments=dict(crossing='formal_scene3_logical_lane_adapter',
                                source_shoulder_and_destination_lanes_preserved=True,
                                other_work_zone_actors_included=False,hesitation_enabled=False,
                                pedestrian_speed_mps=composite.speed),history_equivalent_to_full_run=False)
        if kind=='speed':
            from .speed_fixture import slice_speed_fixture
            route,spec,offset=slice_speed_fixture(route,spec,world.get_map(),carla.Location)
            run_info.update(source_activate_m=source_activate_m,source_entry_m=offset,
                            local_activate_m=spec['activate_m'],preparation_speed_kmh=30,
                            history_equivalent_to_full_run=False)
            if spec['steps'][0].get('keep_lane'):
                from .lane_continuity import trace_lane_corridor
                fixture['steps']['0']=trace_lane_corridor(world.get_map(),route,spec['activate_m'],
                    carla.Location,spec.get('end_route_s_m',route[-1]['distance_m']))
        if kind=='composite':
            from .composite_fixture import CompositeFixture
            composite=CompositeFixture(world,route,spec)
            route,spec,fixture=composite.route,composite.spec,composite.fixture
            run_info.update(source_activate_m=source_activate_m,source_entry_m=composite.offset,
                            local_activate_m=spec['activate_m'],baseline_speed_kmh=30,
                            controller='scripted_composite_acceptance_baseline',
                            fixture_adjustments=dict(crosswalk_polygon_index=composite.crossing['polygon_index'],
                                lane_start_m=composite.lane_start,actor_retirement='after_capture_only',
                                slow_release='after_observed_pedestrian_clear',
                                slow_speed_kmh=20,overtake_speed_kmh=45,walker_speed_mps=1.45))
        if kind=='turn':
            from .turn_fixture import build_turn_route
            route,turn_data=build_turn_route(world.get_map(),route,spec['steps'][0]['direction'],source_activate_m)
            fixture['steps']['0']=turn_data
            spec=dict(spec,activate_m=0)
            run_info.update(source_activate_m=source_activate_m,local_activate_m=0,
                            route_origin='official_map_topology_isolated_turn',baseline_speed_kmh=25)
        if kind=='lane_change':
            from .lane_fixture import select_lane_segment, gap_check
            route,lane_data,target_path,source_entry_m = select_lane_segment(
                world.get_map(),route,spec['steps'][0]['direction'],carla.Location,
                preferred_m=source_activate_m,length_m=seconds*30/3.6+40)
            fixture['steps']['0']=lane_data
            spec=dict(spec,activate_m=0)
            run_info.update(source_activate_m=source_activate_m,source_entry_m=source_entry_m,
                            local_activate_m=0,baseline_speed_kmh=30)
        settings=world.get_settings()
        settings.synchronous_mode=True
        settings.fixed_delta_seconds=.05
        world.apply_settings(settings)
        from .weather import apply_source_weather
        source_config=json.loads((CONFIG_ROOT/catalog.source_file).read_text(encoding='utf-8'))
        run_info['weather']=apply_source_weather(world,source_config,carla.WeatherParameters,advance_world=world.tick)
        p=route[0]
        spawn=carla.Transform(carla.Location(x=p['x'],y=p['y'],z=p['z']+1),carla.Rotation(yaw=p['yaw']))
        entry=world.get_map().get_waypoint(spawn.location)
        if entry is None or entry.is_junction or entry.road_id!=p['road_id'] or entry.lane_id!=p['lane_id']:
            raise ConfigError('recorded route entry does not match live map')
        ego=world.try_spawn_actor(world.get_blueprint_library().find('vehicle.tesla.model3'),spawn)
        if ego is None:
            raise ConfigError('ego spawn unavailable')
        route_manager=SimpleNamespace(route=route,route_length_m=route[-1]['distance_m'],progress_m=0)
        traffic=FixedRouteTraffic(world,client,route_manager,traffic_config)
        if composite is not None:
            composite.spawn(client.get_trafficmanager(traffic.port))
            roles=composite.bindings
        traffic.setup()
        if len(traffic.actors)<3:
            raise ConfigError('insufficient background actors')
        tm=traffic.traffic_manager
        ego.set_autopilot(True,traffic.port)
        tm.auto_lane_change(ego,False)
        speed_issued=kind=='speed' and spec['activate_m']==0
        tm.set_desired_speed(ego,spec['steps'][0]['target_kmh'] if speed_issued else 25 if kind=='turn' else 30)
        tm.set_path(ego,[carla.Location(x=p['x'],y=p['y'],z=p['z']) for p in route[1:101]])
        from .route_window import RouteWindow
        route_window=RouteWindow(route)
        route_end_hold=False
        for sensor_type in ('collision','lane_invasion'):
            bp=world.get_blueprint_library().find('sensor.other.'+sensor_type)
            sensor=world.spawn_actor(bp,carla.Transform(),attach_to=ego)
            sensors.append(sensor)
            if sensor_type=='collision':
                sensor.listen(lambda e:ledger.record('collision',int(e.frame),
                    other_actor_id=e.other_actor.id,other_actor_type=e.other_actor.type_id))
            else:
                def on_lane(e):
                    markings=[str(m.type) for m in e.crossed_lane_markings]
                    ledger.record('lane_invasion',int(e.frame),markings=markings,
                                  illegal=any('Solid' in m or m=='Curb' for m in markings))
                sensor.listen(on_lane)
        # RGB provides a frame heartbeat, not proof that event callbacks are complete.
        bp=world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x','320')
        bp.set_attribute('image_size_y','180')
        bp.set_attribute('sensor_tick','0.0')
        camera=world.spawn_actor(bp,carla.Transform(carla.Location(x=-6,z=3),carla.Rotation(pitch=-10)),attach_to=ego)
        images=queue.Queue(maxsize=8)
        camera.listen(images.put_nowait)
        binding=ActorBinding.from_actor(ego,0)
        from .snapshot_journal import SnapshotJournal
        snapshot_journal=SnapshotJournal(output/'actor_snapshots',{'ego':binding,**roles})
        projector=RouteProjector(route)
        hint=0
        for tick in range(int(seconds/.05)):
            frame=world.tick()
            snapshot=world.get_snapshot()
            if snapshot.frame!=frame:
                raise ConfigError('another client advanced the simulation')
            snapshot_journal.append(snapshot)
            image=images.get(timeout=15)
            if image.frame!=frame:
                raise ConfigError('camera heartbeat frame mismatch')
            if tick%100==0:
                image.save_to_disk(str(output/f'frame_{frame}.png'))
            actor_snapshot=snapshot.find(ego.id)
            if actor_snapshot is None:
                raise ConfigError('ego missing during capture')
            hint=projector.project(actor_snapshot.get_transform().location,hint)['route_s_m']
            v=actor_snapshot.get_velocity()
            motion.append(dict(frame=frame,sim_time_s=snapshot.timestamp.elapsed_seconds,
                               route_s_m=hint,speed_kmh=3.6*(v.x*v.x+v.y*v.y+v.z*v.z)**.5))
            traffic_observation=observe_traffic(snapshot,ego.id,[actor.id for actor in traffic.actors],observation_map)
            motion[-1]['traffic_observation']=traffic_observation
            traffic_density.update(traffic_observation)
            if kind in {'speed','turn','pedestrian'}:
                if route_window.at_end(hint):
                    if not route_end_hold:
                        ego.set_autopilot(False,traffic.port)
                        execution.append(dict(frame=frame,action='route_end_hold',route_s_m=hint))
                    route_end_hold=True
                    ego.apply_control(carla.VehicleControl(throttle=0,brake=1))
                elif tick%20==0:
                    remaining=route_window.points(hint)
                    tm.set_path(ego,[carla.Location(x=p['x'],y=p['y'],z=p['z']) for p in remaining])
                    execution.append(dict(frame=frame,action='route_window_refreshed',
                                          route_s_m=hint,points=len(remaining)))
            if kind=='speed' and not speed_issued and hint>=spec['activate_m']:
                tm.set_desired_speed(ego,spec['steps'][0]['target_kmh'])
                speed_issued=True
                execution.append(dict(frame=frame,route_s_m=hint,action='speed_target_requested',
                                      target_kmh=spec['steps'][0]['target_kmh']))
            if composite is not None:
                motion[-1].update(composite.tick(snapshot,ego,hint,traffic))
            if kind=='lane_change' and not lane_issued and motion[-1]['speed_kmh']>=25:
                decision=gap_check(snapshot,ego,traffic.actors,world.get_map(),lane_data)
                execution.append(dict(frame=frame,**decision))
                if decision['clear']:
                    remaining=[p for p,q in zip(target_path,route) if q['distance_m']>hint+5]
                    tm.set_path(ego,remaining)
                    tm.force_lane_change(ego,lane_data['direction']=='RIGHT')
                    lane_issued=True
                    execution[-1]['action']='lane_change_requested'
            traffic.tick(ego,hint)
        ego.set_autopilot(False,traffic.port)
        camera.stop()
        for sensor in sensors:
            sensor.stop()
        ledger.seal_after_quiet()
        snapshot_journal.close(completed=True)
        collector=SnapshotTruthCollector(spec,route,world.get_map(),binding,roles,fixture,
                                         'isolated_fixture_route')
        with TaskMonitor(collector,output/'assessment') as monitor:
            for snapshot in snapshot_journal.replay(carla.Location):
                monitor.observe(snapshot,ledger.packet(snapshot.frame),
                                dict(frame=snapshot.frame,complete=True,valid=True))
        run_info['task_feedback']=monitor.feedback()
        run_info['safety']=ledger.export()
        run_info['captured_frames']=snapshot_journal.frames
        run_info['final_route_s_m']=hint
        run_info['max_speed_kmh']=max(row['speed_kmh'] for row in motion)
        run_info['traffic']=traffic.snapshot()
        run_info['traffic_density']=traffic_density.result()
        run_info['execution']=execution
        if composite is not None:
            run_info['execution']=composite.events
        run_info['fixture_exercised']=(speed_issued or lane_issued or
                                      (kind=='turn' and hint>=turn_data['junction_end_m']) or
                                      (kind=='pedestrian' and composite.ped_started and composite.seen_conflict) or
                                      (composite is not None and composite.lane_issued))
        run_info['run_valid']=not run_info['safety']['late_events']
        if not run_info['run_valid']:
            run_info['task_feedback']=dict(task_id=spec['task_id'],status='SCENE_INVALID',reason='late_safety_event')
    except Exception as error:
        (output/'run_error.json').write_text(json.dumps(dict(error=type(error).__name__,reason=str(error),
            captured_frames=snapshot_journal.frames if snapshot_journal is not None else 0),indent=2),encoding='utf-8')
        raise
    finally:
        from .cleanup import CleanupJournal
        cleanup=CleanupJournal()
        if snapshot_journal is not None:
            cleanup.attempt('close_snapshot_journal',snapshot_journal.close)
        cleanup.attempt('lifecycle_start',lambda:checkpoint('cleanup_started'))
        def save_motion():
            with (output/'motion.jsonl').open('w',encoding='utf-8') as stream:
                for row in motion:
                    stream.write(json.dumps(row)+'\n')
        cleanup.attempt('save_motion',save_motion)
        for index,sensor in enumerate([camera]+sensors):
            if sensor is not None:
                cleanup.attempt(f'stop_sensor_{index}',lambda s=sensor:s.stop() if s.is_alive and s.is_listening else None)
                cleanup.attempt(f'destroy_sensor_{index}',lambda s=sensor:s.destroy() if s.is_alive else None)
        if composite is not None:
            cleanup.attempt('destroy_composite',composite.destroy)
        if traffic is not None:
            manager=traffic.traffic_manager
            cleanup.attempt('destroy_traffic',traffic.destroy)
            if manager is not None:
                cleanup.attempt('traffic_manager_async',lambda:manager.set_synchronous_mode(False))
                cleanup.attempt('traffic_manager_shutdown',manager.shut_down)
        if ego is not None:
            cleanup.attempt('destroy_ego',lambda:ego.destroy() if ego.is_alive else None)
        cleanup.attempt('restore_world_settings',lambda:world.apply_settings(saved_settings))
        cleanup.attempt('restore_weather',lambda:world.set_weather(saved_weather))
        # Release actor wrappers while their owning client is still alive.
        sensors.clear()
        camera = sensor = ego = traffic = tm = composite = manager = None
        cleanup.attempt('release_wrappers',gc.collect)
        cleanup.attempt('lifecycle_end',lambda:checkpoint('cleanup_attempts_finished'))
        run_info['cleanup']=cleanup.save(output/'cleanup_result.json')
    run_info['cleanup_completed']=run_info['cleanup']['completed']
    run_info['safety']=ledger.export()
    run_info['run_valid']=not run_info['safety']['late_events'] and run_info['cleanup_completed']
    if not run_info['run_valid']:
        run_info['task_feedback']=dict(task_id=spec['task_id'],status='SCENE_INVALID',
                                      reason='cleanup_failed' if not run_info['cleanup_completed'] else 'late_safety_event')
    run_info['episode_safety']=ledger.episode_result()
    (output/'run_result.json').write_text(json.dumps(run_info,ensure_ascii=False,indent=2),encoding='utf-8')
    return run_info


# Preserve callers of the original entry point.
run_speed_fixture = run_fixture
