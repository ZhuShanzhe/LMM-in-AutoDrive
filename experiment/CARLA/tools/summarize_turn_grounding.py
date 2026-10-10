"""Keep concise measured evidence without copying per-frame model memories."""
import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(run):
    risks=Counter();blocked=Counter();targets=Counter();latencies=[]
    count=ready_turns=missing_ready_targets=target_missing_codes=0
    last=None
    for line in (run/'vla_control_decisions.jsonl').open():
        row=json.loads(line);last=row;count+=1
        risks[row['risk_assessment']['risk_level']]+=1
        decision=row['control_decision']
        codes=decision.get('blocked_reason_codes',[]);blocked.update(codes)
        target_missing_codes+=sum('target_location' in c for c in codes)
        if decision.get('decision_status')=='READY' and decision['action'] in ('turn_left','turn_right'):
            ready_turns+=1
            if not decision.get('target_location'):missing_ready_targets+=1
        for item in (row.get('road_step_readiness') or {}).values():
            targets[item['reason']]+=1
        latencies.append(row['full_decision_latency_ms'])
    sorted_latencies=sorted(latencies)
    audit=json.loads((run/'route_command_audit.json').read_text())
    return dict(scope='150s live VLA diagnostic before longitudinal-turn coordination refinement; not full route acceptance',
        decisions=count,last_progress_m=last['route_s_m'],
        final_plan_status=last['control_plan_state']['plan_status'],
        final_step_states=last['control_plan_state']['step_states'],
        ready_turn_commands=ready_turns,missing_ready_turn_targets=missing_ready_targets,
        target_missing_block_codes=target_missing_codes,road_target_reasons=dict(targets),
        risk_levels=dict(risks),blocked_codes=dict(blocked),
        last_traffic_control_status=last['traffic_control_contract']['status'],
        decision_latency_p95_ms=sorted_latencies[int(.95*(len(sorted_latencies)-1))],
        decision_latency_max_ms=max(latencies),latency_scope='decision function; not end-to-end',
        route_task_conflicts=[r for r in audit['records'] if not r['matched']],
        run_outcome=json.loads((run/'benchmark/run_outcome.json').read_text()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    physical=json.loads((args.root/'physical_coordinated_checks.json').read_text())
    result=dict(live_vla=summarize(args.root/'scene2_first_turn/run'),
        geometry_checks=sum(len(r['checks']) for r in physical['geometry']),
        geometry_passed=all(r['passed'] for r in physical['geometry']),
        physical=[{k:v for k,v in r.items() if k!='rows'} for r in physical['physical']],
        physical_scope='fixed-speed non-learned proposal, real ego physics and completion feedback; not learned perception or traffic evaluation')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
