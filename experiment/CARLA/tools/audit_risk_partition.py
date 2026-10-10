"""Audit recorded HIGH stops and replay policy; actor truth is audit-only."""
import argparse
from collections import Counter
import json
from pathlib import Path

from lightweight_vla_adapter.src.risk_behavior import RiskBehaviorPolicy, cautious_permission


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    truth_path=args.run/'frame_ground_truth.jsonl'
    truths={}
    if truth_path.exists():
        with truth_path.open() as stream:
            truths={r['simulation_frame']:r for r in map(json.loads,stream)}
    counts=Counter();old=Counter();new=Counter();policy=RiskBehaviorPolicy()
    evidence=[];distance_values=[];ttcs=[]
    with (args.run/'vla_control_decisions.jsonl').open() as stream:
        for row in map(json.loads,stream):
            if 'risk_assessment' not in row:
                counts['non_decision_records']+=1
                continue
            risk=row['risk_assessment'];previous=risk.get('behavior_policy') or {}
            lidar=previous.get('lidar') or {};timestamp=(risk.get('risk_evidence') or {}).get('decision_timestamp_s')
            counts['frames']+=1;old[previous.get('disposition')]+=1
            replay=policy.apply(risk,lidar,frame_id=risk['frame_id'],timestamp_s=timestamp,
                ego_speed_kmh=row['execution_feedback']['speed_kmh'],parsed_intent=row['parsed_intent'],
                crossing_hazard=bool(previous.get('physical_stop')))
            new[replay['behavior_policy']['disposition']]+=1
            permission=cautious_permission(replay)
            high=risk.get('risk_level')=='high'
            if high:
                counts['high_frames']+=1
                counts['high_with_physical_stop']+=int(bool(previous.get('physical_stop')))
                counts['high_with_new_bounded_permission']+=int(permission is not None)
                counts['held_visual_high_frames']+=int(risk.get('source','').endswith('+risk_continuity'))
            if not (high and previous.get('disposition')=='MUST_STOP'):
                continue
            counts['high_stops']+=1
            truth=truths.get(row['simulation_frame'])
            if truth is None:
                counts['high_stop_without_same_frame_actor_truth']+=1
                continue
            actors=truth.get('actors') or {};counts['high_stop_same_frame_actor_truth']+=1
            close=False
            for actor in actors.values():
                relation=actor.get('relation_to_ego') or {}
                d=relation.get('euclidean_distance_m');ttc=relation.get('time_to_collision_s')
                if isinstance(d,(int,float)):distance_values.append(d)
                if isinstance(ttc,(int,float)):ttcs.append(ttc)
                lateral=abs(relation.get('lateral_m',float('inf')))
                # Conservative local swept-path screen, not a full collision oracle.
                if lateral<=3.5 and (isinstance(d,(int,float)) and d<=previous['required_clearance_m']
                    or isinstance(ttc,(int,float)) and ttc<=5.):close=True
            counts['high_stop_with_known_actor_near_path_hazard']+=int(close)
            counts['high_stop_cut_in_still_armed']+=int(truth.get('runtime_state',{}).get('cut_in_phase')=='ARMED')
            if len(evidence)<8:
                evidence.append(dict(simulation_frame=row['simulation_frame'],route_s_m=row['route_s_m'],
                    source=risk.get('source'),reason_codes=risk.get('reason_codes'),
                    physical_stop=previous.get('physical_stop'),lidar_clear_m=lidar.get('clear_distance_m'),
                    lidar_obstacle_m=lidar.get('nearest_obstacle_m'),
                    runtime_state=truth.get('runtime_state'),
                    actors={k:{'relation':v.get('relation_to_ego'),'lane':v.get('lane')} for k,v in actors.items()}))
    result=dict(scope='recorded fragment; actor-role coverage is not an all-world safety certificate',
        policy_replay_scope='same recorded sensor inputs; no closed-loop success or model accuracy claim',
        truth_used_by_policy=False,scheduled_risk_labels_used_as_truth=False,
        counts=dict(counts),previous_dispositions=dict(old),replayed_dispositions=dict(new),
        known_actor_distance_range_m=[min(distance_values),max(distance_values)] if distance_values else None,
        known_actor_ttc_min_s=min(ttcs) if ttcs else None,samples=evidence)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
