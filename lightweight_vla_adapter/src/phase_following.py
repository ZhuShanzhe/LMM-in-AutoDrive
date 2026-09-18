"""Phase-balanced supervision on measured recovery trajectories, not runtime rules."""

import torch
from torch.nn import functional as F
from .joint_longitudinal_loss import trajectory_displacement

VERSION='phase_balanced_following/1.0'
PHASE_NAMES=('catch_up','settle','follow','hold_stop','restart','cruise')


def phase_labels(gap, ego, lead, valid):
    target=6.+1.4*ego
    band=torch.maximum(torch.full_like(target,3.),target*.25)
    error=gap-target
    closing=ego-lead
    labels=torch.full_like(gap,2,dtype=torch.long)
    labels=torch.where(error>band,0,labels)
    labels=torch.where((error < -band) | ((closing>.7)&(error<=band)),1,labels)
    labels=torch.where((ego<1.)&(lead>.8)&(gap>6.2),4,labels)
    labels=torch.where((ego<.5)&(lead<.3)&(gap<9.),3,labels)
    return torch.where(valid.bool(),labels,5)


def phase_terms(out,batch):
    speed=out['speed_sequence_mps']
    ego=batch['initial_speed_mps'][:,0].to(speed)
    lead0=batch['lead_initial_mps'].to(speed)
    lead=batch['lead_future_mps'].to(speed)
    gap0=batch['initial_gap_m'].to(speed)
    valid=batch['following_valid'].to(speed).bool()
    desired=batch['desired_speed_mps'].to(speed)
    labels=phase_labels(gap0,ego,lead0,valid)
    gap=gap0[:,None]+trajectory_displacement(lead,lead0)-trajectory_displacement(speed,ego)
    # A fixed per-window distance reference cannot be gamed by changing predicted speed.
    reference_gap=6.+1.4*torch.minimum(desired,lead0.clamp_min(0))
    tolerance=torch.maximum(torch.full_like(reference_gap,3.),reference_gap*.25)
    reference_relative=(.35*(gap0-reference_gap)).clamp(-3.,3.)
    times=torch.arange(1,31,device=speed.device,dtype=speed.dtype)*.1
    desired_path=(lead+reference_relative[:,None]).clamp_min(0)
    desired_path=torch.minimum(desired_path,desired[:,None])
    # Reachable progress target respects acceleration and speed limits from the current state.
    desired_path=torch.minimum(desired_path,ego[:,None]+1.2*times)
    desired_path=torch.maximum(desired_path,(ego[:,None]-2.*times).clamp_min(0))
    desired_path=torch.minimum(desired_path,torch.maximum(desired,ego)[:,None])
    progress_shortfall=(trajectory_displacement(desired_path,ego)-trajectory_displacement(speed,ego)).relu()
    catch=(F.smooth_l1_loss((desired_path-speed).relu(),torch.zeros_like(speed),reduction='none')
        +F.smooth_l1_loss(progress_shortfall/3.,torch.zeros_like(speed),reduction='none')).mean(1)
    distance_error=gap-reference_gap[:,None]
    relative=speed-lead
    settle_relative=(.35*distance_error).clamp(-2.,2.)
    settle=(F.smooth_l1_loss(relative,settle_relative,reduction='none')+
        F.smooth_l1_loss((-distance_error-tolerance[:,None]).relu()/3.,torch.zeros_like(speed),reduction='none')).mean(1)
    follow=(F.smooth_l1_loss((distance_error.abs()-tolerance[:,None]).relu()/3.,torch.zeros_like(speed),reduction='none')
        +F.smooth_l1_loss(relative,settle_relative,reduction='none')).mean(1)
    stopped=lead<.3
    stop=((speed.square()+out['commanded_acceleration_mps2'].relu().square())*stopped).sum(1)/stopped.sum(1).clamp_min(1)
    teacher=batch['speed_target'][:,0].to(speed)
    imitation=F.smooth_l1_loss(speed/.8,teacher/.8,reduction='none').mean(1)
    phase_loss=torch.stack((catch,settle,follow,stop,imitation+catch,imitation),1).gather(1,labels[:,None]).squeeze(1)
    closing=relative.relu()
    reserve=gap-2.-.5*closing-closing.square()/4.
    risk=F.smooth_l1_loss((-reserve).relu(),torch.zeros_like(speed),reduction='none').mean(1)*valid
    unsafe=(gap.amin(1)<2.)&valid
    stop_eligible=(labels==3)&stopped[:,0]
    stop_accelerating=stop_eligible&(out['acceleration_sequence_mps2'][:,0]>.3)
    means=[phase_loss[labels==i].mean() for i in range(len(PHASE_NAMES)) if bool((labels==i).any())]
    balanced=torch.stack(means).mean()
    loss=2.*balanced+3.*risk.sum()/valid.sum().clamp_min(1)
    return loss,dict(labels=labels,phase_loss=phase_loss,risk=risk,valid=valid,
        unsafe=unsafe,stop_eligible=stop_eligible,stop_accelerating=stop_accelerating)


def summarize_phase_batches(batches):
    joined={key:torch.cat([b[key].detach().cpu() for b in batches]) for key in batches[0]}
    per_phase={}
    for i,name in enumerate(PHASE_NAMES):
        mask=joined['labels']==i
        per_phase[name]=dict(samples=int(mask.sum()),loss=float(joined['phase_loss'][mask].mean()) if mask.any() else None)
    losses=[p['loss'] for p in per_phase.values() if p['samples']]
    return dict(phases=per_phase,balanced_phase_loss=sum(losses)/len(losses),
        unsafe_plan_rate=float(joined['unsafe'].sum()/joined['valid'].sum().clamp_min(1)),
        stop_acceleration_rate=float(joined['stop_accelerating'].sum()/joined['stop_eligible'].sum().clamp_min(1)),
        stop_windows=int(joined['stop_eligible'].sum()))


def selection_key(metrics):
    p=metrics['phase_metrics']
    return (p['unsafe_plan_rate'],p['stop_acceleration_rate'],p['balanced_phase_loss'])
