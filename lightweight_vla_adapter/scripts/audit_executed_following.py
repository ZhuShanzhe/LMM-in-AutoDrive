"""Audit fixed-window physical following and distinguish routine operations from safety overrides."""

import argparse
import json
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from lightweight_vla_adapter.src.following_distance import audit_spacing


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();report=json.loads((args.run/'report.json').read_text());result=[]
    required_interval=float(report.get('sequence_execution',{}).get('operation_interval_s',1.))
    for episode in report['episodes']:
        truth=[json.loads(x) for x in (args.run/(episode['id']+'_truth.jsonl')).read_text().splitlines()]
        end=20. if episode['mode']=='stopgo' else 30.
        begin=10. if episode['mode']=='stopgo' else 15.
        if episode['mode']=='brake_close':begin,end=24.,39.
        rows=[r for r in truth if begin<=r['t_s']<end]
        speed=np.array([r['speed_kmh'] for r in rows])
        ref=np.array([episode['desired'] if episode['mode']=='cruise' else r['lead_speed_kmh'] for r in rows])
        fraction=float(np.mean(abs(speed-ref)<=3.)) if len(rows) else None
        logs=[];path=args.run/(episode['id']+'_runtime.jsonl')
        if path.exists():logs=[json.loads(x) for x in path.read_text().splitlines()]
        execution=[x.get('risk_assessment',{}).get('sequence_execution',{}) for x in logs]
        regular=[x['operation_time_s'] for x in execution if x.get('operation_updated') and not x.get('safety_bypass')]
        violation=sum(b-a<required_interval-1e-5 for a,b in zip(regular,regular[1:]))
        gaps=[r['gap_m'] for r in truth if r['gap_m'] is not None]
        gap_rows=[r for r in truth if r.get('decision_gap_injected') and 18.35<=r['t_s']<18.5]
        sustained=[r for r in truth if r['t_s']>=15. and
            (episode['mode']!='stopgo' or r['t_s']<22. or r['t_s']>=38.) and
            (episode['mode']!='brake_close' or r['t_s']>=24.)]
        sustained_errors=[abs(r['speed_kmh']-(episode['desired'] if episode['mode']=='cruise' else r['lead_speed_kmh']))
                          for r in sustained]
        steady_fraction=float(np.mean(np.array(sustained_errors)<=3.)) if sustained else None
        expired_max=max([r['throttle'] for r in gap_rows]) if gap_rows else None
        minimum_ttc=[]
        initial=[episode['initial_state']] if episode.get('initial_state') else []
        for r in initial+truth:
            closing=(r['speed_kmh']-r['lead_speed_kmh'])/3.6
            if r['gap_m'] is not None and closing>.5:minimum_ttc.append(r['gap_m']/closing)
        unsafe=[r for r in truth if r['gap_m'] is not None and r['gap_m']<2. and r['speed_kmh']>1.]
        by_frame={r['frame']:r for r in truth}
        if episode.get('initial_state'):
            r=episode['initial_state'];by_frame[r['frame']]=r
        critical=[]
        for log in logs:
            r=by_frame.get(log.get('simulation_frame'))
            if r is None or r['gap_m'] is None:continue
            closing=(r['speed_kmh']-r['lead_speed_kmh'])/3.6
            imminent=(r['gap_m']<2. and r['speed_kmh']>1.) or (closing>.5 and r['gap_m']/closing<2.)
            if imminent:critical.append(log)
        raw_positive=sum(x.get('risk_assessment',{}).get('event_memory',{}).get('longitudinal_sequence',{}).get('acceleration_mps2',[0.])[0]>.3 for x in critical)
        spacing=audit_spacing(truth,collision_events=episode['collision_events']) if episode['mode'] in ('follow18','follow26') else dict(status='not_applicable_to_constant_lead_following')
        spacing.pop('phase_rows',None)
        result.append(dict(episode=episode['id'],mode=episode['mode'],window_s=[begin,end],frames=len(rows),
            spacing_assessment=spacing,
            speed_within_3kmh_fraction=fraction,speed_status='diagnostic_only',
            speed_p5_p95_kmh=np.percentile(speed,[5,95]).tolist() if len(speed) else [],
            mean_reference_speed_kmh=float(ref.mean()) if len(ref) else None,
            min_bumper_gap_m=min(gaps) if gaps else None,collision_events=episode['collision_events'],
            routine_operation_updates=len(regular),routine_interval_violations=violation,required_routine_interval_s=required_interval,
            minimum_routine_interval_s=min(np.diff(regular)) if len(regular)>1 else None,
            safety_bypass_updates=sum(bool(x.get('safety_bypass')) for x in execution),
            sequence_forwarded_decisions=sum(bool(x.get('forwarded')) for x in execution),
            runtime_fallbacks=episode.get('runtime',{}).get('fallback_count'),
            min_ttc_when_closing_s=min(minimum_ttc) if minimum_ttc else None,
            moving_frames_with_gap_under_2m=len(unsafe),
            physical_brake_frames=sum(r['brake']>.05 for r in truth),
            model_output_applied_count=sum(bool(r.get('model_output_applied')) for r in logs),
            gated_or_rewritten_count=sum(not bool(r.get('model_output_applied')) for r in logs),
            independently_critical_decisions=len(critical),
            raw_positive_acceleration_on_critical_decisions=raw_positive,
            final_braking_on_critical_decisions=sum(x.get('control_decision',{}).get('action') in ('decelerate','stop','emergency_brake') for x in critical),
            expired_interval_throttle_max=expired_max,
            expiry_fault_status=('not_reached' if expired_max is None else 'pass' if expired_max<=1e-6 else 'fail') if episode.get('fault') else 'not_injected',
            sustained_frames=len(sustained),sustained_within_3kmh_fraction=steady_fraction,
            sustained_speed_status='diagnostic_only',
            full_decision_latency_ms=episode.get('runtime',{}).get('full_decision_latency_ms'),
            sensor_to_decision_response_ms=episode.get('runtime',{}).get('sensor_to_decision_response_ms'),
            progress_m=episode['progress_m']))
    out=dict(scope='CARLA episodes; fixed-window comparison plus sustained measurement from 15s to episode end; stop-go excludes 22..38s transient',
        collection_status=report['status'],
        criterion=f'Constant-lead following uses provisional spacing acquisition/maintenance checks; speed fractions are diagnostic only; routine interval at least {required_interval}s; urgent responses exempt',
        speed_fraction_used_for_acceptance=False,
        caveats=['Routine plan updates are not low-level actuator sampling or an independent neural policy score.',
                 'Passing speed stability does not waive collision, unsafe following or sensor faults.',
                 'Independent critical condition: moving with bumper gap <2m, or closing >0.5m/s with TTC <2s. Zero critical decisions is missing coverage, not 100% neural safety.'],episodes=result)
    total_frames=sum(r['sustained_frames'] for r in result)
    out['overall']=dict(completed_episodes=len(result),actual_simulation_seconds=sum(e['frames'] for e in report['episodes'])*.05,
        provisional_spacing_check_pass_episodes=sum(r['spacing_assessment']['status']=='meets_provisional_spacing_checks' for r in result),
        sustained_frames=total_frames,
        weighted_sustained_within_3kmh_fraction=sum((r['sustained_within_3kmh_fraction'] or 0.)*r['sustained_frames'] for r in result)/total_frames if total_frames else None,
        collision_events=sum(r['collision_events'] for r in result),
        runtime_fallbacks=sum(r['runtime_fallbacks'] or 0 for r in result),
        routine_interval_violations=sum(r['routine_interval_violations'] for r in result),required_routine_interval_s=required_interval,
        fault_checks={status:sum(r['expiry_fault_status']==status for r in result) for status in ('pass','fail','not_reached','not_injected')},
        independently_critical_decisions=sum(r['independently_critical_decisions'] for r in result),
        raw_positive_acceleration_on_critical_decisions=sum(r['raw_positive_acceleration_on_critical_decisions'] for r in result))
    (args.run/'following_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))


if __name__=='__main__':main()
