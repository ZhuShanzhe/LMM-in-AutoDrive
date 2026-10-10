"""Replay recorded risk/radar packets, not counterfactual vehicle trajectories."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiment/CARLA'))
from control.risk_continuity import RiskContinuityFilter


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();filt=RiskContinuityFilter()
    raw_counts=Counter();effective_counts=Counter();reasons=Counter()
    prev_raw=prev_effective=None;raw_changes=effective_changes=emergency_inputs=delayed_emergencies=0
    changed_samples=[];count=0
    for line in args.input.open():
        row=json.loads(line);count+=1;raw=row['risk_assessment']
        result,diag=filt.apply(raw,timestamp_s=(raw.get('risk_evidence') or {}).get('decision_timestamp_s'),
            frame_id=f'carla_{row["simulation_frame"]}',tracked_observation=row.get('tracked_observation'))
        raw_counts[raw['risk_level']]+=1;effective_counts[result['risk_level']]+=1;reasons[diag['reason']]+=1
        raw_changes+=prev_raw is not None and prev_raw!=raw['risk_level']
        effective_changes+=prev_effective is not None and prev_effective!=result['risk_level']
        prev_raw=raw['risk_level'];prev_effective=result['risk_level']
        if raw['recommended_action']=='emergency_brake':
            emergency_inputs+=1;delayed_emergencies+=result['recommended_action']!='emergency_brake'
        if diag['changed'] and len(changed_samples)<30:
            changed_samples.append(dict(frame=row['simulation_frame'],route_s_m=row['route_s_m'],
                raw_level=raw['risk_level'],diagnostic=diag))
    report=dict(scope='fixed recorded risk/radar packets; not a new trajectory or task-success evaluation',
        tracker_identity_changes_not_available_in_old_record=True,frames=count,
        raw_counts=dict(raw_counts),effective_counts=dict(effective_counts),
        raw_transitions=raw_changes,effective_transitions=effective_changes,
        transition_reduction=1.-effective_changes/raw_changes if raw_changes else 0.,
        emergency_inputs=emergency_inputs,delayed_emergencies=delayed_emergencies,
        reasons=dict(reasons),changed_samples=changed_samples)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='changed_samples'},indent=2))


if __name__=='__main__':main()
