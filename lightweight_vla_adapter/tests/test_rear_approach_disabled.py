import json
from pathlib import Path

import pytest
import torch

from lightweight_vla_adapter.src.sequence_policy import SequenceEventHead, SequenceMemoryRuntime
from lightweight_vla_adapter.tests.test_prediction_contract import base_output


@pytest.mark.parametrize('front_high',[False,True])
def test_joint_rear_residual_cannot_raise_or_lower_the_base_forward_risk(front_high):
    torch.manual_seed(12)
    head=SequenceEventHead().eval()
    with torch.no_grad():
        head.risk_head.weight.zero_();head.risk_head.bias[:]=torch.tensor([-10.,-10.,20.])
    runtime=SequenceMemoryRuntime(head).eval()
    base=base_output()
    if front_high:base.visual_risk_logits[:]=torch.tensor([[-5.,-5.,10.]])
    memory=torch.zeros(1,80,52);memory[:,-1,42]=.5
    batch=dict(event_memory=memory,event_memory_valid=torch.ones(1,80,dtype=torch.bool))
    with torch.inference_mode():result=runtime(base,batch,longitudinal_authorized=torch.tensor([True]))
    torch.testing.assert_close(result.visual_risk_logits,base.visual_risk_logits)
    assert runtime.diagnostics['raw_risk_probabilities'][2]>.99
    assert runtime.diagnostics['risk_residual_applied'] is False
    assert runtime.diagnostics['event_probabilities'][-1]==0.
    assert runtime.diagnostics['disabled_events']==['rear_approach']


def test_rear_motion_cannot_change_the_deployed_sequence_or_action():
    torch.manual_seed(13)
    runtime=SequenceMemoryRuntime(SequenceEventHead()).eval()
    base=base_output();a=torch.zeros(1,80,52);a[:,-1,42]=.5
    b=a.clone();b[:,:,10:20]=.9;b[:,:,30:40]=1.;b[:,:,49]=.7;b[:,:,51]=1.
    valid=torch.ones(1,80,dtype=torch.bool)
    with torch.inference_mode():
        first=runtime(base,dict(event_memory=a,event_memory_valid=valid),longitudinal_authorized=torch.tensor([True]))
        speed=list(runtime.diagnostics['longitudinal_sequence']['speed_mps'])
        second=runtime(base,dict(event_memory=b,event_memory_valid=valid),longitudinal_authorized=torch.tensor([True]))
    torch.testing.assert_close(first.action_logits,second.action_logits)
    assert speed==runtime.diagnostics['longitudinal_sequence']['speed_mps']


def test_deployed_configs_disable_rear_approach_and_following_sources():
    root=Path(__file__).resolve().parents[2]
    for name in ['challenge_signal_generalization','universal_three_scene_v6_sensor_policy']:
        config=json.loads((root/'lightweight_vla_adapter/configs'/f'{name}.json').read_text())
        assert config['rear_approach_risk_enabled'] is False
    for name in ['basic_voice_urban_5km','scene_2_town05_runtime','scene_3_emergency_6km_runtime']:
        config=json.loads((root/'experiment/CARLA/configs'/f'{name}.json').read_text())['traffic']
        assert config['rear_following_distance_m']>=60.
        assert config.get('following_source_actor_reserve',config.get('following_source_max_actors'))==0
