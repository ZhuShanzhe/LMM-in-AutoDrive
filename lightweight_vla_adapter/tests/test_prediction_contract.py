from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lightweight_vla_adapter.src.contracts import ACTION_LABELS
from lightweight_vla_adapter.src.decision_adapter import AdapterOutput, validate_prediction_output
from lightweight_vla_adapter.src.event_memory import EventMemoryHead
from lightweight_vla_adapter.src.sequence_policy import SequenceEventHead, SequenceMemoryRuntime
from lightweight_vla_adapter.src.pipeline import decode_visual_risk_assessment, LightweightVLAPipeline


def base_output():
    logits=torch.zeros(1,len(ACTION_LABELS))
    logits[0,ACTION_LABELS.index('keep_lane')]=5
    return AdapterOutput(action_logits=logits,target_speed_kmh=torch.tensor([30.]),
        target_lane_logits=torch.zeros(1,3),target_pointer_logits=torch.zeros(1,1),
        confidence_logits=torch.zeros(1),confidence=torch.tensor([.7]),
        decision_embedding=torch.zeros(1,256),visual_risk_logits=torch.tensor([[5.,0.,0.]]),
        risk_input_features=torch.zeros(1,1))


@pytest.mark.parametrize('authorized,available',[(False,False),(False,True),(True,False),(True,True)])
def test_sequence_published_only_when_actually_applied(authorized,available):
    torch.manual_seed(91)
    runtime=SequenceMemoryRuntime(SequenceEventHead()).eval()
    memory=torch.zeros(1,80,52)
    memory[:,-1,40]=.25
    memory[:,-1,42]=.5
    valid=torch.zeros(1,80,dtype=torch.bool)
    valid[:,-1]=available
    base=base_output()
    with torch.inference_mode():
        result=runtime(base,dict(event_memory=memory,event_memory_valid=valid),longitudinal_authorized=torch.tensor([authorized]))
    applied=authorized and available
    assert runtime.diagnostics['applied']==applied
    assert runtime.diagnostics['authorized']==authorized
    assert runtime.diagnostics['available']==available
    assert ('longitudinal_sequence' in runtime.diagnostics)==applied
    if not applied:
        assert result is base


def test_sequence_does_not_leak_from_previous_valid_prediction():
    runtime=SequenceMemoryRuntime(SequenceEventHead()).eval()
    memory=torch.zeros(1,80,52)
    valid=torch.ones(1,80,dtype=torch.bool)
    with torch.inference_mode():
        runtime(base_output(),dict(event_memory=memory,event_memory_valid=valid),longitudinal_authorized=torch.tensor([True]))
        assert 'longitudinal_sequence' in runtime.diagnostics
        valid[:]=False
        runtime(base_output(),dict(event_memory=memory,event_memory_valid=valid),longitudinal_authorized=torch.tensor([True]))
    assert 'longitudinal_sequence' not in runtime.diagnostics


def test_invalid_history_nan_is_ignored_but_valid_nan_rejected():
    head=EventMemoryHead().eval()
    x=torch.zeros(1,80,52)
    valid=torch.ones(1,80,dtype=torch.bool)
    valid[:,:40]=False
    with torch.inference_mode():
        original=head(x,valid)
        x[:,:40]=float('nan')
        masked=head(x,valid)
        for key in original:
            torch.testing.assert_close(original[key],masked[key])
        x[:,-1,0]=float('nan')
        with pytest.raises(ValueError,match='Non-finite valid event memory'):
            head(x,valid)


def test_unavailable_nan_sequence_has_finite_diagnostics():
    import json
    runtime=SequenceMemoryRuntime(SequenceEventHead()).eval()
    memory=torch.full((1,80,52),float('nan'))
    with torch.inference_mode():
        runtime(base_output(),dict(event_memory=memory,event_memory_valid=torch.zeros(1,80,dtype=torch.bool)),longitudinal_authorized=torch.tensor([True]))
    json.dumps(runtime.diagnostics,allow_nan=False)
    assert not runtime.diagnostics['applied']


@pytest.mark.parametrize('direct',[False,True])
def test_layered_runtime_masks_nan_in_both_histories(direct):
    import json
    from lightweight_vla_adapter.src.layered_context import LayeredSequenceHead
    from lightweight_vla_adapter.src.behavior_memory import SUMMARY_DIM,SUMMARY_SLOTS
    runtime=SequenceMemoryRuntime(LayeredSequenceHead(direct_sequence=direct)).eval()
    batch=dict(event_memory=torch.full((1,80,52),float('nan')),
        event_memory_valid=torch.zeros(1,80,dtype=torch.bool),
        behavior_memory=torch.full((1,SUMMARY_SLOTS,SUMMARY_DIM),float('nan')),
        behavior_memory_valid=torch.zeros(1,SUMMARY_SLOTS,dtype=torch.bool),
        recursive_state=torch.zeros(1,6,32),layered_context=torch.zeros(1,48))
    base=base_output()
    with torch.inference_mode():
        result=runtime(base,batch,longitudinal_authorized=torch.tensor([True]))
    assert result is base
    json.dumps(runtime.diagnostics,allow_nan=False)
    assert not runtime.diagnostics['applied']
    assert 'longitudinal_sequence' not in runtime.diagnostics


def test_joint_risk_can_legitimately_supervise_rear_escape_acceleration():
    from lightweight_vla_adapter.src.counterfactual_motion_targets import rollout_targets,ACTION_NAMES
    targets=rollout_targets([[10,0,150,10,0,15,20,0,.85,20],
                             [10,0,15,0,0,1000,10,0,.85,20]])
    assert targets['risk'].tolist()==[2,2]
    assert ACTION_NAMES[targets['action'][0]]=='accelerate'
    assert ACTION_NAMES[targets['action'][1]]=='emergency_brake'
    assert targets['horizon'][0,0].max()==0
    assert targets['horizon'][0,1].max()==1


@pytest.mark.parametrize('name',['action_logits','target_speed_kmh','target_lane_logits','target_pointer_logits','confidence','visual_risk_logits'])
@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_bad_output_cannot_decode_to_apparently_valid_action(name,bad):
    output=base_output()
    getattr(output,name).reshape(-1)[0]=bad
    with pytest.raises(ValueError,match='Non-finite model output'):
        validate_prediction_output(output)


@pytest.mark.parametrize('field',['risk_score','risk_uncertainty','risk_horizon_logits'])
def test_invalid_auxiliary_risk_rejected(field):
    with pytest.raises(ValueError,match='Non-finite visual risk output'):
        decode_visual_risk_assessment(torch.zeros(1,3),**{field:torch.tensor([float('nan')])})


def test_stage_provenance_does_not_replace_base_with_residual():
    class Model(torch.nn.Module):
        use_temporal_risk=False
        use_candidate_entities=False
        def __init__(self):
            super().__init__()
            self.anchor=torch.nn.Parameter(torch.zeros(1))
        def forward(self,**kwargs):
            return base_output()
    class Residual:
        diagnostics={'applied':True,'horizon_seconds':[1,2,4],
            'front_rear_horizon_scores':[[0.,0.,0.],[0.,1.,1.]],
            'longitudinal_sequence':{'schema_version':'longitudinal_sequence/1.0','acceleration_mps2':[1.]}}
        def __call__(self,output,motion,**kwargs):
            logits=output.action_logits.clone().zero_()
            logits[0,ACTION_LABELS.index('accelerate')]=8
            return replace(output,action_logits=logits,target_speed_kmh=torch.tensor([31.]),visual_risk_logits=torch.tensor([[0.,0.,5.]]))
    pipeline=LightweightVLAPipeline(Model(),device='cpu',checkpoint_loaded=True)
    batch=SimpleNamespace(validate=lambda:None,camera_bev=torch.zeros(1,8,1,1))
    pipeline._move=lambda b:b
    pipeline._model_inputs=lambda b:{}
    result=pipeline.predict_proposal(batch,request_id='test',frame_id='frame1',candidate_entity_ids=[[]],
        decision_residual=Residual(),motion_inputs={'motion_values':torch.zeros(1,1),'motion_valid_mask':torch.zeros(1,1)},longitudinal_authorized=True)
    provenance=pipeline.last_visual_risk_assessment['prediction_provenance']
    assert provenance['base']['action']=='keep_lane'
    assert provenance['effective']['action']==result['action']=='accelerate'
    assert provenance['effective_action_source']=='sequence_first_sample_encoding'
    assert provenance['residual_applied'] is True
    assert pipeline.last_visual_risk_assessment['risk_level']=='high'
    semantics=pipeline.last_visual_risk_assessment['risk_semantics']
    assert semantics['scope']=='joint_front_rear_rollout_surrogate'
    assert semantics['directional_horizon_scores']['rear']==[0.,1.,1.]
    assert semantics['calibrated'] is False
