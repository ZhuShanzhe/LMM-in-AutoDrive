"""Optional exact model-boundary sampling for reproducible offline trials."""
import hashlib
import json
from pathlib import Path
import shutil
import time
import math
import numpy as np


INPUTS=('camera_bev','lidar_bev','ego_features','candidate_features','candidate_mask',
        'intent_tokens','intent_mask','camera_images','camera_view_mask','environment_features')


class BenchmarkTensorCapture:
    def __init__(self, root, *, interval_s=5., progress_step_m=10., max_records=1200,
                 max_bytes=2*1024**3, min_free_bytes=6*1024**3):
        if interval_s<=0 or progress_step_m<=0 or max_records<=0 or max_bytes<=0:
            raise ValueError('Invalid capture limits')
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=False)
        self.interval_s=float(interval_s);self.progress_step_m=float(progress_step_m)
        self.max_records=int(max_records);self.max_bytes=int(max_bytes);self.min_free_bytes=int(min_free_bytes)
        self.last_time=None;self.last_progress=None;self.last_event=None;self.seen=set()
        self.count=0;self.bytes=0;self.halted=False;self.errors=[]
        self.stream=(self.root/'manifest.jsonl').open('w',encoding='utf-8')

    def record(self,batch,*,sensor_frame,timestamp_s,route_s_m,context,decision):
        if self.halted or self.count>=self.max_records or sensor_frame in self.seen:return
        if (not isinstance(sensor_frame,int) or isinstance(sensor_frame,bool)
                or sensor_frame<0 or not math.isfinite(timestamp_s) or not math.isfinite(route_s_m)):
            raise ValueError('Capture requires a finite actual sensor-frame context')
        event=tuple(context.get('sampling_event_key',(context.get('request_id'),context.get('step_id'))))
        due=(self.last_time is None or timestamp_s-self.last_time>=self.interval_s
             or abs(route_s_m-self.last_progress)>=self.progress_step_m or event!=self.last_event)
        if not due:return
        start=time.perf_counter()
        temporary=None
        try:
            values=batch if isinstance(batch,dict) else {key:getattr(batch,key,None) for key in INPUTS}
            arrays={key:values[key].detach().cpu().numpy() for key in INPUTS
                    if values.get(key) is not None}
            if not arrays or any(not np.isfinite(a).all() for a in arrays.values()):
                raise ValueError('Missing or nonfinite model inputs')
            size=sum(a.nbytes for a in arrays.values())
            if self.bytes+size>self.max_bytes or shutil.disk_usage(self.root).free-size<self.min_free_bytes:
                self.halted=True;self.errors.append(dict(reason='storage_budget',frame=sensor_frame));return
            name=f'{self.count:05d}_{sensor_frame}.npz';path=self.root/name
            temporary=path.with_suffix('.tmp')
            with temporary.open('wb') as f:np.savez_compressed(f,**arrays)
            digest=hashlib.sha256(temporary.read_bytes()).hexdigest();temporary.replace(path)
            row=dict(schema_version='model_boundary_capture/1.0',sample_index=self.count,
                sensor_frame=int(sensor_frame),timestamp_s=float(timestamp_s),route_s_m=float(route_s_m),
                path=name,sha256=digest,context=context,decision=decision,
                inputs={k:dict(shape=list(a.shape),dtype=str(a.dtype)) for k,a in arrays.items()},
                capture_wall_ms=(time.perf_counter()-start)*1000,
                scope='Exact actual model inputs, not independent labels or a recurrent-state checkpoint')
            self.stream.write(json.dumps(row,allow_nan=False,ensure_ascii=False)+'\n');self.stream.flush()
            self.count+=1;self.bytes+=size;self.seen.add(sensor_frame)
            self.last_time=timestamp_s;self.last_progress=route_s_m;self.last_event=event
        except Exception as error:
            self.errors.append(dict(frame=sensor_frame,error_type=type(error).__name__,message=str(error)))
            self.halted=True
            if temporary is not None and temporary.exists():temporary.unlink()

    def close(self):
        self.stream.close()
        result=dict(schema_version='model_boundary_capture_summary/1.0',records=self.count,
            uncompressed_tensor_bytes=self.bytes,halted=self.halted,errors=self.errors,
            scope='Sampled exact model boundary; labels and complete episode evidence are separate',
            precision_note='No float rounding or uint8 conversion performed',
            instrumentation_note='Capture cost reported per record; do not treat instrumented run as uninstrumented latency')
        (self.root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
