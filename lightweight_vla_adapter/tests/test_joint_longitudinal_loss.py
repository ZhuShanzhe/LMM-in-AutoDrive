import pytest
import torch

from lightweight_vla_adapter.src.joint_longitudinal_loss import (
    joint_longitudinal_loss, joint_longitudinal_metrics, normal_hold_projection,
    trajectory_displacement, LOSS_WEIGHTS,
)
from lightweight_vla_adapter.src.sequence_execution import SequenceExecutionPolicy
from lightweight_vla_adapter.src.sequence_policy import integrate_acceleration


def sample(acceleration=0.):
    command=torch.full((2,30),float(acceleration),requires_grad=True)
    initial=torch.full((2,),10.)
    speed,a=integrate_acceleration(command,initial,torch.full((2,),30.))
    out=dict(speed_sequence_mps=speed,acceleration_sequence_mps2=a,
             commanded_acceleration_mps2=command,risk_logits=torch.zeros(2,3,requires_grad=True))
    batch=dict(initial_speed_mps=initial[:,None].expand(-1,2),desired_speed_mps=torch.full((2,),30.),
               speed_target=speed.detach()[:,None].expand(-1,2,-1),
               accel_target=a.detach()[:,None].expand(-1,2,-1))
    return out,batch,command


def test_exact_cruise_all_terms_zero():
    out,batch,_=sample()
    loss,terms=joint_longitudinal_loss(out,out,batch,out)
    assert set(terms)==set(LOSS_WEIGHTS)
    assert abs(float(loss.detach()))<1e-6


def test_acceleration_and_speed_errors_backpropagate():
    out,batch,command=sample()
    batch['speed_target']=batch['speed_target']+1.
    batch['accel_target']=batch['accel_target']+1.
    loss,terms=joint_longitudinal_loss(out,out,batch,out)
    loss.backward()
    assert terms['acceleration']>0 and terms['held_speed']>0 and terms['displacement']>0
    assert torch.isfinite(command.grad).all() and command.grad.abs().sum()>0


def test_emergency_targets_not_forced_into_one_second_hold():
    out,batch,_=sample()
    batch['accel_target']=torch.full((2,2,30),-6.)
    loss,terms=joint_longitudinal_loss(out,out,batch,out)
    assert terms['held_speed']==0 and terms['held_acceleration']==0
    assert terms['first_acceleration']>0
    assert torch.isfinite(loss)


@pytest.mark.parametrize('initial,desired,acceleration',[(10.,30.,1.),(10.,10.2,2.),(.2,10.,-1.)])
def test_normal_projection_matches_runtime_operation_boundary(initial,desired,acceleration):
    sequence=dict(schema_version='longitudinal_sequence/1.0',dt_s=.1,
                  speed_mps=[initial]*30,acceleration_mps2=[acceleration]*30)
    speed,held=normal_hold_projection(torch.tensor([[acceleration]*30]),torch.tensor([initial]),torch.tensor([desired]))
    executor=SequenceExecutionPolicy()
    actual=[]
    for i in range(10):
        proposal=executor.update({},sequence,timestamp_s=i*.1,episode_id='test',speed_kmh=initial*3.6,
                                 desired_speed_kmh=desired*3.6,risk={'risk_level':'low'})
        actual.append(proposal['target_speed_kmh']/3.6)
    torch.testing.assert_close(speed[0],torch.tensor(actual),atol=1e-5,rtol=1e-5)
    assert float(held[0])==pytest.approx(acceleration)


def test_distance_uses_trapezoidal_measured_speed_integration():
    speed=torch.tensor([[1.,2.,3.]])
    distance=trajectory_displacement(speed,torch.tensor([0.]))
    torch.testing.assert_close(distance,torch.tensor([[.05,.2,.45]]))


def test_jerk_is_change_per_second_not_raw_acceleration_difference():
    out,batch,_=sample()
    batch['accel_target']=torch.arange(30).float()[None,None,:].expand(2,2,-1)*.1
    metrics=joint_longitudinal_metrics(out,batch)
    torch.testing.assert_close(metrics['jerk_mae_mps3'],torch.ones(2))
