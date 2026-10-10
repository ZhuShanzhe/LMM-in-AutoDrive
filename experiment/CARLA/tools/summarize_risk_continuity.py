"""Summarize measured live risk continuity and task outcomes separately."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raw_counts=Counter();effective_counts=Counter();blocked=Counter();reasons=Counter()
    previous_raw=previous_effective=None;raw_changes=effective_changes=count=0
    high_frames=delayed_high_frames=missing_targets=ready_turns=0
    latencies=[];last=None
    for line in (args.run/'vla_control_decisions.jsonl').open():
        row=json.loads(line);last=row;count+=1
        diag=row['risk_temporal_continuity'];risk=row['risk_assessment'];decision=row['control_decision']
        raw=diag['raw_level'];effective=risk['risk_level']
        raw_counts[raw]+=1;effective_counts[effective]+=1;reasons[diag['reason']]+=1
        raw_changes+=previous_raw is not None and previous_raw!=raw
        effective_changes+=previous_effective is not None and previous_effective!=effective
        previous_raw=raw;previous_effective=effective
        if raw=='high':
            high_frames+=1;delayed_high_frames+=risk['recommended_action']!='emergency_brake'
        blocked.update(decision.get('blocked_reason_codes',[]))
        if decision['decision_status']=='READY' and decision['action'] in ('turn_left','turn_right'):
            ready_turns+=1;missing_targets+=not bool(decision.get('target_location'))
        latencies.append(row['full_decision_latency_ms'])
    route_audit=json.loads((args.run/'route_command_audit.json').read_text())
    benchmark=json.loads((args.run/'benchmark/summary.json').read_text())
    summary=json.loads((args.run/'summary.json').read_text())
    result=dict(scope='live x86 camera/lidar/radar/text-to-control diagnostic; selected task1, not full three-route acceptance',
        frames=count,last_progress_m=last['route_s_m'],final_plan_status=last['control_plan_state']['plan_status'],
        final_step_states=last['control_plan_state']['step_states'],
        raw_risk_counts=dict(raw_counts),effective_risk_counts=dict(effective_counts),
        raw_transitions=raw_changes,effective_transitions=effective_changes,
        transition_reduction=1.-effective_changes/raw_changes if raw_changes else 0.,
        raw_high_frames=high_frames,delayed_high_frames=delayed_high_frames,
        continuity_reasons=dict(reasons),blocked_codes=dict(blocked),
        ready_turn_commands=ready_turns,missing_ready_turn_targets=missing_targets,
        final_risk_continuity=last['risk_temporal_continuity'],
        final_traffic_status=last['traffic_control_contract']['status'],
        final_raw_probabilities=last['risk_assessment'].get('learned_probabilities',last['risk_assessment'].get('probabilities')),
        decision_latency_p95_ms=sorted(latencies)[int(.95*(len(latencies)-1))],
        decision_latency_max_ms=max(latencies),latency_scope='decision function; not end-to-end latency',
        benchmark_success_count=benchmark['success_count'],benchmark_total_tasks=benchmark['total_tasks'],
        benchmark_safety=benchmark['episode_safety'],
        route_conflicts=[r for r in route_audit['records'] if not r['matched']],
        runtime_summary=summary,
        run_outcome=json.loads((args.run/'benchmark/run_outcome.json').read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('runtime_summary','route_conflicts','final_step_states')},indent=2))


if __name__=='__main__':main()
