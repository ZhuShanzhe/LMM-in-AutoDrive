import hashlib
import json
import numpy as np
import pytest
import torch
from control.benchmark_tensor_capture import BenchmarkTensorCapture


def make_capture(tmp_path,**kwargs):
    return BenchmarkTensorCapture(tmp_path/'capture',min_free_bytes=0,**kwargs)


def record(capture,frame=1,time=1.,distance=1.,**kwargs):
    return capture.record(kwargs.get('batch',{'ego_features':torch.tensor([[1.25,2.5]])}),
        sensor_frame=frame,timestamp_s=time,route_s_m=distance,
        context=kwargs.get('context',dict(request_id='r',step_id='s')),decision={'action':'keep_lane'})


def test_exact_values_and_hash(tmp_path):
    c=make_capture(tmp_path);record(c);summary=c.close()
    assert summary['records']==1
    row=json.loads((c.root/'manifest.jsonl').read_text())
    path=c.root/row['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
    np.testing.assert_array_equal(np.load(path)['ego_features'],[[1.25,2.5]])
    assert row['inputs']['ego_features']['dtype']=='float32'


def test_sampling_deduplicates_actual_frame(tmp_path):
    c=make_capture(tmp_path);record(c);record(c);record(c,2,1.1,1.2)
    assert c.count==1
    record(c,3,7.,1.3);assert c.count==2;c.close()


def test_event_transition_sampled(tmp_path):
    c=make_capture(tmp_path);record(c)
    record(c,2,1.1,1.1,context={'request_id':'r','step_id':'next'})
    assert c.count==2;c.close()


def test_ephemeral_transport_ids_do_not_exhaust_sampling(tmp_path):
    c=make_capture(tmp_path)
    for frame in range(1,21):
        record(c,frame,frame*.05,1.,context={'request_id':f'unified-{frame}',
            'step_id':'step_1','sampling_event_key':['Continue safely','step_1']})
    assert c.count==1;c.close()


def test_budget_halts_without_partial_file(tmp_path):
    c=make_capture(tmp_path,max_bytes=1);record(c)
    assert c.halted and c.count==0 and not list(c.root.glob('*.npz'));c.close()


@pytest.mark.parametrize('batch',[None,{}, {'ego_features':torch.tensor([[float('nan')]])}])
def test_invalid_tensor_capture_does_not_raise_to_control(tmp_path,batch):
    c=make_capture(tmp_path);record(c,batch=batch)
    assert c.halted and c.errors and c.count==0;c.close()


@pytest.mark.parametrize('kwargs',[{'frame':True},{'frame':-1},{'time':float('nan')},{'distance':float('inf')}])
def test_invalid_frame_context_rejected(tmp_path,kwargs):
    c=make_capture(tmp_path)
    with pytest.raises(ValueError):record(c,**kwargs)
    c.close()
