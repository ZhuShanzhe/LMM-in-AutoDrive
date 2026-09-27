"""Audit model-stage provenance without treating risk scores as truth labels."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def audit(path):
    counts=Counter()
    violations=[]
    last={}
    def finite(value):
        if isinstance(value,float):
            return math.isfinite(value)
        if isinstance(value,dict):
            return all(finite(v) for v in value.values())
        if isinstance(value,list):
            return all(finite(v) for v in value)
        return True
    with Path(path).open(encoding='utf-8') as stream:
        for line_no,line in enumerate(stream,1):
            row=json.loads(line)
            counts['records']+=1
            if 'error' in row:
                counts['error_records']+=1
                continue
            counts['decisions']+=1
            last=row
            risk=row.get('risk_assessment') or {}
            memory=risk.get('event_memory') or {}
            provenance=risk.get('prediction_provenance')
            issues=[]
            if provenance is None:
                counts['missing_provenance']+=1
                issues.append('missing_provenance')
            else:
                counts['with_provenance']+=1
                if not finite(provenance):
                    issues.append('nonfinite_provenance')
                for stage in ('base','effective'):
                    prediction=provenance.get(stage) or {}
                    probabilities=prediction.get('risk_probabilities') or []
                    if len(probabilities)!=3:
                        issues.append(stage+'_invalid_risk_shape')
                    elif probabilities[2]>=.55 and probabilities[2]==max(probabilities) and prediction.get('action')=='accelerate':
                        counts[stage+'_high_risk_accelerate']+=1
            if memory:
                expected=memory.get('authorized') is True and memory.get('available') is True
                if memory.get('applied') is not expected:
                    issues.append('incorrect_memory_applied_flag')
                if 'longitudinal_sequence' in memory and not expected:
                    issues.append('sequence_without_applied_memory')
            if not finite(row.get('network_proposal_before_execution')) or not finite(row.get('control_decision')):
                issues.append('nonfinite_proposal_or_control')
            if issues:
                violations.append(dict(line=line_no,frame=row.get('simulation_frame'),issues=issues))
    return dict(scope='model contract and self-consistency only, not risk accuracy or driving acceptance',
                counts=dict(counts),contract_violations=violations,
                last_route_s_m=last.get('route_s_m'),last_dispatch=last.get('command_dispatch'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=audit(args.log)
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(1 if result['contract_violations'] or not result['counts'].get('decisions') else 0)
