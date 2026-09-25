"""Reconstruct a task from full-episode snapshots without changing original results."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from .catalog import ConfigError, number
from .episode import unpack_snapshot
from .monitor import TaskMonitor
from .safety_events import SafetyLedger
from .truth_capture import ActorBinding, SnapshotTruthCollector


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lane_diagnostics(rows, spec, fixture):
    results=[]
    for index,step in enumerate(spec['steps']):
        if step['kind']!='lane_change':
            continue
        lane=fixture['steps'][str(index)]
        targets=lane.get('target_lane_keys',[lane['target_lane_key']])
        target_frames=0
        longest=0.0
        since=previous=None
        for row in rows:
            ego=row.get('ego',{})
            now=row.get('sim_time_s')
            eligible=row.get('scenario_valid') is True and ego.get('route_s_m',-1)>=step.get('start_route_s_m',spec['activate_m'])
            target=eligible and ego.get('lane_key') in targets
            target_frames+=int(target)
            stable=(target and not ego['in_junction'] and
                    abs(ego['lateral_error_m'])<=step['max_lateral_error_m'] and
                    abs(ego['heading_error_deg'])<=step['max_heading_error_deg'])
            if not stable:
                since=None
            else:
                if since is None or previous is None or now-previous>spec['max_frame_gap_s']+1e-8:
                    since=now
                longest=max(longest,now-since)
            previous=now
        results.append(dict(step_index=index,target_lane_frames=target_frames,
            longest_stable_s=longest,required_hold_s=step['hold_s'],
            scope='geometric_condition_duration_not_sequential_task_success'))
    return results


def overtake_diagnostics(rows, spec, result):
    """Explain ordered passing evidence without overriding the task oracle."""
    evidence=result.get('evidence',[])
    start=next((e for e in evidence if e['event']=='START'),None)
    reports=[]
    for index,step in enumerate(spec['steps']):
        if step['kind']!='overtake':
            continue
        activation=start if index==0 else next((e for e in evidence
            if e['event']=='STEP_SUCCESS' and e['step']==index-1),None)
        first_ahead=first_passed=None
        ahead_after=False
        first_gap_after=None
        for row in rows:
            if row.get('scenario_valid') is not True:
                continue
            ego=row['ego']
            target=row.get('actors',{}).get(step['target_role'])
            if target is None or target.get('route_corridor_id')!=ego.get('route_corridor_id'):
                continue
            ahead=target['route_s_m']>ego['route_s_m']
            gap=ego['rear_route_s_m']-target['front_route_s_m']
            stamp=dict(frame=row['frame'],sim_time_s=row['sim_time_s'])
            if ahead and first_ahead is None:
                first_ahead=stamp
            if gap>=step['rear_clearance_m'] and first_passed is None:
                first_passed=stamp
            if activation is not None and row['frame']>activation['frame']:
                ahead_after |= ahead
                if first_gap_after is None:
                    first_gap_after=gap
        reports.append(dict(step_index=index,target_role=step['target_role'],
            previous_step_completed=activation,first_target_ahead=first_ahead,
            first_clearance_reached=first_passed,target_ahead_after_step_activation=ahead_after,
            rear_clearance_at_first_step_sample_m=first_gap_after,
            scope='ordering_diagnostic_not_task_success'))
    return reports


def step_timing_diagnostics(rows,spec,result):
    """Account for observed duration, without guessing why the vehicle stopped."""
    start=next((e for e in result.get('evidence',[]) if e['event']=='START'),None)
    if start is None:
        return []
    reports=[]
    for index,step in enumerate(spec['steps']):
        end=next((e for e in result['evidence']
                  if e['event']=='STEP_SUCCESS' and e['step']==index),None)
        window=[r for r in rows if r['frame']>=start['frame'] and
                (end is None or r['frame']<=end['frame'])]
        low=covered=0.0
        for a,b in zip(window,window[1:]):
            dt=b['sim_time_s']-a['sim_time_s']
            if (a.get('scenario_valid') is not True or b.get('scenario_valid') is not True or
                    not 0<dt<=spec['max_frame_gap_s']+1e-8):
                continue
            covered+=dt
            if 0<=a['ego']['speed_kmh']<=0.5 and 0<=b['ego']['speed_kmh']<=0.5:
                low+=dt
        reports.append(dict(step_index=index,kind=step['kind'],completed=end is not None,
            observed_duration_s=window[-1]['sim_time_s']-window[0]['sim_time_s'] if window else 0,
            covered_interval_s=covered,near_stopped_s=low,
            stop_cause='not_recorded',scope='observation_timing_not_model_latency'))
        if end is None:
            break
        start=end
    return reports


def reassess(source, task_id, output, world_map, location_factory, timeout_s=None,
             rebuild_speed_corridor=False):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output==source or output.is_relative_to(source):
        raise ConfigError('diagnostic output must be outside the original capture')
    task=(source/'tasks'/task_id).resolve()
    if task.parent!=source/'tasks':
        raise ConfigError('invalid task path')
    manifest=read_json(task/'manifest.json')
    episode=read_json(source/'summary.json')
    if world_map.name.split('/')[-1]!=episode['map_name'].split('/')[-1]:
        raise ConfigError('diagnostic map name mismatch')
    map_xml=world_map.to_opendrive() if hasattr(world_map,'to_opendrive') else None
    map_hash=hashlib.sha256(map_xml.encode('utf-8')).hexdigest() if map_xml else None
    original_map_hash=episode.get('map_geometry_sha256')
    if original_map_hash:
        if map_hash!=original_map_hash:
            raise ConfigError('diagnostic map geometry mismatch')
        stored=source/'map.xodr'
        if not stored.is_file() or hashlib.sha256(stored.read_bytes()).hexdigest()!=original_map_hash:
            raise ConfigError('captured map geometry missing or changed')
    if manifest['source_sha256']!=episode['source_sha256']:
        raise ConfigError('task and episode source hashes differ')
    captured={}
    for name in ('spec','fixture','route'):
        raw=(task/f'{name}.json').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=manifest[f'{name}_sha256']:
            raise ConfigError(f'captured {name} hash mismatch')
        captured[name]=json.loads(raw)
    if captured['spec']!=manifest['spec'] or captured['fixture']!=manifest['fixture']:
        raise ConfigError('task manifest differs from stored evidence')
    spec=copy.deepcopy(captured['spec'])
    if timeout_s is not None:
        if number(timeout_s,'diagnostic timeout')<=0:
            raise ConfigError('diagnostic timeout must be positive')
        spec['timeout_s']=timeout_s
    safety=episode['safety']
    if not safety['sealed'] or safety['late_events']:
        raise ConfigError('source safety ledger incomplete')
    ledger=SafetyLedger()
    for event in safety['events']:
        ledger.record(**event)
    ledger.sealed=True  # Reuse the recorded finalization, not a new transport guarantee.
    bindings={role:ActorBinding(**value) for role,value in manifest['bindings'].items()}
    fixture=copy.deepcopy(captured['fixture'])
    if rebuild_speed_corridor:
        if not original_map_hash:
            raise ConfigError('rebuilding speed corridor requires captured map geometry')
        from .lane_continuity import trace_lane_corridor
        for i,step in enumerate(spec['steps']):
            if step['kind']=='speed' and step.get('keep_lane'):
                fixture['steps'][str(i)]=trace_lane_corridor(world_map,
                    captured['route'],spec['activate_m'],location_factory,
                    spec.get('end_route_s_m',captured['route'][-1]['distance_m']))
    collector=SnapshotTruthCollector(spec,captured['route'],world_map,bindings['ego'],
        {k:v for k,v in bindings.items() if k!='ego'},fixture,'formal_route')
    with (task/'task_truth.jsonl').open(encoding='utf-8') as stream:
        first_frame=json.loads(next(stream))['frame']
    output.mkdir(parents=True,exist_ok=False)
    with TaskMonitor(collector,output/'assessment') as monitor:
        with (source/'episode_truth.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line)
                if row['frame']<first_frame:
                    continue
                snapshot=unpack_snapshot(row,location_factory)
                invalid=[role for role in bindings if role!='ego' and
                         row.get('roles',{}).get(role,{}).get('status')!='BOUND']
                monitor.observe(snapshot,ledger.packet(snapshot.frame),
                    dict(frame=snapshot.frame,complete=True,valid=not invalid,reason=str(invalid)))
    with (source/'episode_truth.jsonl').open('rb') as stream:
        journal_hash=hashlib.file_digest(stream,'sha256').hexdigest()
    with (output/'assessment/task_truth.jsonl').open(encoding='utf-8') as stream:
        observations=[json.loads(line) for line in stream]
    report=dict(scope='diagnostic_only_not_formal_acceptance',benchmark_ready=False,
        original_result=read_json(task/'task_result.json'),diagnostic_result=monitor.feedback(),
        original_timeout_s=captured['spec']['timeout_s'],diagnostic_timeout_s=spec['timeout_s'],
        source_journal_sha256=journal_hash,map_name=world_map.name,
        diagnostic_map_sha256=map_hash,
        original_map_geometry_hash_available=bool(original_map_hash),
        rebuilt_speed_corridor=rebuild_speed_corridor,
        lane_diagnostics=lane_diagnostics(observations,spec,captured['fixture']),
        overtake_diagnostics=overtake_diagnostics(observations,spec,monitor.oracle.result()),
        step_timing=step_timing_diagnostics(observations,spec,monitor.oracle.result()),
        limitations=([] if original_map_hash else ['original_capture_has_no_map_geometry_hash'])+
                    ['recorded_collision_and_solid_markings_only'])
    (output/'diagnostic.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--task',required=True)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--timeout-s',type=float)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=2000)
    args=parser.parse_args()
    from carla_bootstrap import setup_carla_api
    setup_carla_api()
    import carla
    client=carla.Client(args.host,args.port)
    client.set_timeout(30)
    result=reassess(args.source,args.task,args.output,client.get_world().get_map(),carla.Location,args.timeout_s)
    print(json.dumps({k:result[k] for k in ('scope','original_timeout_s','diagnostic_timeout_s','diagnostic_result')}))


if __name__=='__main__':
    main()
