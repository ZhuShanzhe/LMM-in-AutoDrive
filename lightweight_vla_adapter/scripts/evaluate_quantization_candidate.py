"""Labelled FP32/BC comparison in an isolated Horizon x86 environment."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--float-onnx',required=True,type=Path)
    parser.add_argument('--bc',required=True,type=Path)
    parser.add_argument('--validation',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--limit',type=int,default=224)
    args=parser.parse_args()
    if args.output.exists() or args.limit<1:parser.error('New output and positive limit required')
    import numpy as np
    import onnxruntime as ort
    from horizon_tc_ui.hb_runtime import HBRuntime
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    reference=ort.InferenceSession(str(args.float_onnx),sess_options=options,providers=['CPUExecutionProvider'])
    quantized=HBRuntime(str(args.bc))
    names=[node.name for node in reference.get_inputs()]
    outputs=[node.name for node in reference.get_outputs()]
    results=[]
    samples=sorted(args.validation.glob('input_*.npz'))[:args.limit]
    if not samples:raise ValueError('No validation inputs')
    for sample in samples:
        with np.load(sample,allow_pickle=False) as data:feeds={name:data[name].copy() for name in names}
        start=time.perf_counter();fp=dict(zip(outputs,reference.run(None,feeds)));fp_ms=(time.perf_counter()-start)*1000
        start=time.perf_counter();q=dict(zip(quantized.output_names,quantized.run(quantized.output_names,feeds)))
        q_ms=(time.perf_counter()-start)*1000
        if any(not np.isfinite(value).all() for value in q.values()):raise ValueError('Nonfinite BC output')
        row=dict(path=sample.name,sha256=hashlib.sha256(sample.read_bytes()).hexdigest(),
            float_cpu_ms=fp_ms,bc_functional_x86_ms=q_ms,
            errors={name:float(np.abs(q[name]-fp[name]).max()) for name in outputs},
            agreement={name:bool(np.array_equal(fp[name].argmax(-1),q[name].argmax(-1)))
                for name in outputs if name.endswith('logits')},labelled=False)
        label=sample.with_name(sample.name.replace('input_','labels_'))
        if label.exists():
            with np.load(label,allow_pickle=False) as labels:
                row['labelled']=True;row['label_sha256']=hashlib.sha256(label.read_bytes()).hexdigest()
                row['accuracy']={}
                for head,key in (('action_logits','action_label'),('visual_risk_logits','risk_label'),('lane_logits','lane_label')):
                    truth=int(labels[key].reshape(-1)[0])
                    row['accuracy'][head]=dict(label=truth,float_correct=int(fp[head].argmax(-1)[0])==truth,
                        quantized_correct=int(q[head].argmax(-1)[0])==truth)
                truth=float(labels['speed_label'].reshape(-1)[0])
                row['speed_mae']={name:abs(float(value['speed_kmh'].reshape(-1)[0])-truth)
                    for name,value in (('float',fp),('quantized',q))}
        results.append(row)
        if len(results)%16==0:print('validated',len(results),flush=True)
    labelled=[row for row in results if row['labelled']];metrics={}
    for head in ('action_logits','visual_risk_logits','lane_logits'):
        f=float(np.mean([row['accuracy'][head]['float_correct'] for row in labelled])) if labelled else None
        q=float(np.mean([row['accuracy'][head]['quantized_correct'] for row in labelled])) if labelled else None
        metrics[head]=dict(labelled_n=len(labelled),float_accuracy=f,quantized_accuracy=q,
            relative_degradation=(f-q)/f if f else None,
            agreement=float(np.mean([row['agreement'][head] for row in results])))
    high=[row for row in labelled if row['accuracy']['visual_risk_logits']['label']==2]
    metrics['high_risk_recall']={name:float(np.mean([row['accuracy']['visual_risk_logits'][key] for row in high]))
        if high else None for name,key in (('float','float_correct'),('quantized','quantized_correct'))}
    metrics['high_risk_recall']['n']=len(high)
    metrics['speed_mae']={name:float(np.mean([row['speed_mae'][name] for row in labelled]))
        if labelled else None for name in ('float','quantized')}
    report=dict(scope='Caller-provided labels; BC x86 functional inference, not board timing or full-chain acceptance',
        input_count=len(results),labelled_count=len(labelled),metrics=metrics,records=results,
        float_onnx_sha256=hashlib.sha256(args.float_onnx.read_bytes()).hexdigest(),
        bc_sha256=hashlib.sha256(args.bc.read_bytes()).hexdigest(),
        validation_source=str(args.validation.resolve()),board_tested=False,closed_loop_tested=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
