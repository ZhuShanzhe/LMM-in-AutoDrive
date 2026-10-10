"""Audit a recorded run under explicitly changed engineering criteria."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from benchmark.safety_adjusted_assessment import SafetyAdjustedTaskOracle
from benchmark.task_oracle import TERMINAL


def assess(run):
    records={}
    with (run/'vla_control_decisions.jsonl').open()as stream:
        for line in stream:
            row=json.loads(line)
            if 'risk_assessment'not in row:continue
            records[row['simulation_frame']]={k:deepcopy(row.get(k))for k in
                ('control_decision','traffic_control_contract','previous_pid_execution')}
            risk=row['risk_assessment']
            records[row['simulation_frame']]['risk']={k:risk.get(k)for k in
                ('frame_id','risk_evidence','behavior_policy')}
            radar=risk.get('physical_forward_radar')or{}
            records[row['simulation_frame']]['risk']['physical_forward_radar']={k:radar.get(k)for k in
                ('nearest_distance_m','nearest_closing_velocity_mps','nearest_closing_distance_m')}
    results={}
    for task in sorted((run/'benchmark/tasks').iterdir()):
        if not (task/'spec.json').exists():continue
        oracle=SafetyAdjustedTaskOracle(json.loads((task/'spec.json').read_text()))
        with (task/'task_truth.jsonl').open()as stream:
            for line in stream:
                row=json.loads(line);row['execution_evidence']=records.get(row['frame'],{})
                result=oracle.update(row)
                if result['status']in TERMINAL:break
            else:result=oracle.end_of_stream()
        results[task.name]=dict(engineering_result=result,
            original_strict_result=json.loads((task/'task_result.json').read_text()))
    return dict(schema_version='safety_adjusted_assessment/1.0',
        scope='changed engineering criteria; not competition strict acceptance',tasks=results,
        original_recordings_modified=False)


def validate_output(run,output):
    run=run.resolve();output=output.resolve()
    if output.is_relative_to(run) and output!=run/'engineering_assessment.json':
        raise ValueError('Inside the recording directory, only engineering_assessment.json may be written')
    if output.exists():
        try:schema=json.loads(output.read_text()).get('schema_version')
        except (ValueError,AttributeError):schema=None
        if schema!='safety_adjusted_assessment/1.0':
            raise ValueError('Refusing to replace an existing non-engineering recording or document')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    try:validate_output(args.run,args.output)
    except ValueError as error:parser.error(str(error))
    result=assess(args.run)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for task,entry in result['tasks'].items():
        r=entry['engineering_result']
        print(task,r['status'],'strict:',entry['original_strict_result']['status'])
    return 0


if __name__=='__main__':raise SystemExit(main())
