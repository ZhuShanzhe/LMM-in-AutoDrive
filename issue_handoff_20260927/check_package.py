"""Validate handoff evidence; known findings are not acceptance passes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner')
    parser.add_argument('--decode-capture',action='store_true')
    parser.add_argument('--probe-contracts',action='store_true')
    args=parser.parse_args()
    manifest=read(HERE/'SHA256SUMS.json')
    errors=[name for name,expected in manifest.items() if not (HERE/name).is_file() or sha(HERE/name)!=expected]
    if errors:
        raise SystemExit('Evidence hash mismatch: '+str(errors))
    people=[p for p in HERE.iterdir() if (p/'issues.json').exists() and (not args.owner or p.name==args.owner)]
    if not people:
        raise SystemExit('Unknown owner')
    result={'evidence_integrity':'PASS','files':len(manifest),'acceptance_status':'NOT_RUN','owners':[]}
    for p in people:
        spec=read(p/'issues.json')
        result['owners'].append(dict(owner=p.name,issues=[dict(id=x['id'],title=x['title'],status='OPEN_FOR_REVIEW_OR_FIX') for x in spec['issues']]))
    changed=[name for name,expected in read(HERE/'_common/runtime_source_sha256.json').items() if sha(ROOT/name)!=expected]
    result['runtime_changed_since_diagnostic']=changed
    result['scene2_baseline']=read(HERE/'_common/scene2/decision_summary.json')['counts']
    result['geometry_mismatches']=[r['task'] for r in read(HERE/'_common/scene2_geometry.json')['rows'] if r['status']=='MISMATCH']
    if args.decode_capture:
        sys.path[:0]=[str(ROOT),str(ROOT/'experiment/CARLA')]
        from evaluation.model_rig_replay import ModelRigReplayDataset,decode_sensor
        with tempfile.TemporaryDirectory(prefix='handoff-replay-') as tmp:
            target=Path(tmp).resolve()
            with zipfile.ZipFile(HERE/'_common/capture_subset.zip') as archive:
                for name in archive.namelist():
                    if not (target/name).resolve().is_relative_to(target):
                        raise ValueError('Unsafe archive member')
                archive.extractall(target)
            for name,expected in read(target/'raw_sha256.json').items():
                assert sha(target/name)==expected,name
            data=ModelRigReplayDataset(target)
            shapes=[]
            for frame in data:
                shapes.append(dict(frame=frame.simulation_frame,sensors={name:list(decode_sensor(frame,name).shape) for name in frame.artifacts}))
            result['capture']=dict(frames=len(data),decoded=shapes,input_manifest=data.integrity_manifest(),scope='decoding/integrity only, not model accuracy or exact temporal replay')
    if args.probe_contracts:
        sys.path[:0]=[str(ROOT),str(ROOT/'experiment/CARLA'),str(HERE)]
        from probe_contracts import probes
        result['current_contract_probes']=probes()
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
