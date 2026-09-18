"""Shared causal observation/event features. Never read actor truth or future labels."""

import math
import numpy as np
import torch
from torch import nn
from .behavior_memory import BehaviorSequenceHead
from .sequence_policy import integrate_acceleration,first_action
from .contracts import ACTION_LABELS
from torch.nn import functional as F

CONTEXT_VERSION = 'layered_observation_event/1.0'
LAYERED_SCHEMA = 'layered_behavior_sequence/1.0'
DIRECT_LAYERED_SCHEMA = 'layered_behavior_sequence/2.0'
FEATURE_NAMES = (
    'ego_speed', 'ego_acceleration', 'ego_yaw_rate', 'command_speed',
    'target_available', 'target_measured', 'target_predicted', 'radar_fresh',
    'gap', 'closing_speed', 'target_speed', 'target_radial_acceleration',
    'position_uncertainty', 'measurement_age', 'lateral_offset', 'closing_risk',
    'gap_error', 'short_headway', 'nearest_available', 'nearest_gap',
    'nearest_closing', 'nearest_risk', 'invalid_route', 'unknown',
    'confirmed_count', 'tentative_count', 'observed_count', 'observer_error',
    'ego_ax', 'ego_ay', 'target_in_corridor', 'nearest_differs',
    'event_active', 'event_count', 'behavior_accelerate', 'behavior_decelerate',
    'behavior_keep', 'behavior_stop', 'behavior_emergency', 'event_age',
    'transition_count', 'transition_age', 'target_history_present', 'target_history_age',
    'target_reference_present', 'history_count', 'event_goal_speed', 'event_context_available',
)


def encode_layered_context(observation, events, desired_speed_kmh, now_s):
    if not math.isfinite(float(now_s)):raise ValueError('Finite context timestamp required')
    o=observation or {};events=events or {};ego=o.get('ego',{})
    values=np.zeros(len(FEATURE_NAMES),np.float32)
    def put(name,value,scale=1.):
        x=float(value)
        if not math.isfinite(x):raise ValueError(f'Nonfinite layered feature: {name}')
        values[FEATURE_NAMES.index(name)]=np.clip(x/scale,-5.,5.)
    vx=float(ego.get('vx_mps',0));vy=float(ego.get('vy_mps',0));speed=math.hypot(vx,vy)
    ax=float(ego.get('ax_mps2',0));ay=float(ego.get('ay_mps2',0))
    accel=(vx*ax+vy*ay)/speed if speed>.1 else 0.
    for name,value,scale in [('ego_speed',speed,40),('ego_acceleration',accel,8),
        ('ego_yaw_rate',ego.get('yaw_rate_dps',0),90),('command_speed',desired_speed_kmh,100),
        ('ego_ax',ax,8),('ego_ay',ay,8)]:put(name,value,scale)
    put('radar_fresh',bool(o.get('radar_fresh')))
    put('invalid_route',bool(o.get('invalid_route')))
    put('unknown',o.get('status','UNKNOWN')=='UNKNOWN')
    put('observer_error',bool(o.get('observer_error')))
    target=o.get('selected') or {};nearest=o.get('nearest_obstacle') or {}
    if target and target.get('status') in ('TRACKED','PREDICTED') and 0<=target.get('age_s',31)<=.5:
        gap=target.get('route_gap_m');closing=target.get('closing_speed_mps')
        if gap is not None and closing is not None:
            put('target_available',1);put('target_measured',target['status']=='TRACKED')
            put('target_predicted',target['status']=='PREDICTED')
            for name,value,scale in [('gap',gap,80),('closing_speed',closing,20),
                ('target_speed',speed-closing,40),('target_radial_acceleration',target.get('radial_acceleration_mps2',0),8),
                ('position_uncertainty',target.get('position_std_m',0),5),('measurement_age',target.get('age_s',0),.5),
                ('lateral_offset',target.get('route_lateral_m',0),4),('closing_risk',max(0.,closing)/max(1.,gap),1),
                ('gap_error',gap-6.-1.4*speed,40),('short_headway',max(0.,6.+1.4*speed-gap),30)]:put(name,value,scale)
            put('target_in_corridor',bool(target.get('in_route_corridor')))
    if (nearest and nearest.get('route_gap_m') is not None
            and nearest.get('status') in ('TRACKED','PREDICTED') and 0<=nearest.get('age_s',31)<=.5):
        gap=nearest['route_gap_m'];closing=nearest.get('closing_speed_mps') or 0.
        put('nearest_available',1);put('nearest_gap',gap,80);put('nearest_closing',closing,20)
        put('nearest_risk',max(0.,closing)/max(1.,gap))
        put('nearest_differs',nearest.get('track_id')!=target.get('track_id'))
    entities=o.get('entities',[])
    put('confirmed_count',sum(e.get('status')!='TENTATIVE' for e in entities),64)
    put('tentative_count',sum(e.get('status')=='TENTATIVE' for e in entities),64)
    put('observed_count',sum(e.get('status')=='TRACKED' for e in entities),64)
    active=events.get('active_events',[]);transitions=events.get('recent_transitions',[])
    put('event_active',bool(active));put('event_count',len(active),16)
    if active:
        event=active[-1];behavior=str(event.get('current_behavior','')).upper()
        if behavior=='HOLD':behavior='KEEP_LANE'
        for label,name in [('ACCELERATE','accelerate'),('DECELERATE','decelerate'),('KEEP_LANE','keep'),
            ('STOP','stop'),('EMERGENCY_BRAKE','emergency')]:put('behavior_'+name,behavior==label)
        put('event_age',max(0.,now_s-event.get('started_s',now_s)),60)
        put('target_reference_present',bool((event.get('boundary_observation') or {}).get('target_ref')))
    put('transition_count',len(transitions),32)
    if transitions:put('transition_age',max(0.,now_s-transitions[-1].get('timestamp_s',now_s)),60)
    history=o.get('target_history',{}).get('entries',[])
    history=[h for h in history if 0<=now_s-h['last_seen_s']<=30.000001]
    recent=next((h for h in history if h['track_id']==target.get('track_id')),None)
    put('target_history_present',recent is not None)
    if recent:put('target_history_age',now_s-recent['last_seen_s'],30)
    put('history_count',len(history),8192)
    put('event_goal_speed',desired_speed_kmh,100);put('event_context_available',bool(events))
    return values


class LayeredSequenceHead(BehaviorSequenceHead):
    schema_version=LAYERED_SCHEMA
    def __init__(self,direct_sequence=False):
        super().__init__()
        self.direct_sequence=direct_sequence
        if direct_sequence:
            self.schema_version=DIRECT_LAYERED_SCHEMA
            self.layered_sequence=nn.Sequential(nn.Linear(448,128),nn.ReLU(),nn.Linear(128,30))
            nn.init.zeros_(self.layered_sequence[-1].weight);nn.init.zeros_(self.layered_sequence[-1].bias)
        self.layered_observation=nn.Sequential(nn.Linear(32,128),nn.ReLU())
        self.layered_events=nn.Sequential(nn.Linear(16,64),nn.ReLU())
        self.layered_gate=nn.Sequential(nn.Linear(256+192,64),nn.ReLU(),nn.Linear(64,2))
        self.layered_projection=nn.Linear(192,256)
        nn.init.zeros_(self.layered_projection.weight);nn.init.zeros_(self.layered_projection.bias)
        self.force_no_layered_context=False
        self.force_no_event_context=False

    def forward(self,memory,valid,context=None,*,layered_context=None,**kwargs):
        if layered_context is None or layered_context.shape!=(memory.shape[0],len(FEATURE_NAMES)):
            raise ValueError('Layered checkpoint requires versioned [B,48] observation/event context')
        if not torch.isfinite(layered_context).all():raise ValueError('Nonfinite layered context')
        if context is None:context=memory.new_zeros((len(memory),256))
        observation=self.layered_observation(layered_context[:,:32])
        events=self.layered_events(layered_context[:,32:])
        if self.force_no_event_context:events=torch.zeros_like(events)
        gate=self.layered_gate(torch.cat((context,observation,events),-1)).sigmoid()
        residual=self.layered_projection(torch.cat((observation*gate[:,:1],events*gate[:,1:]),-1))
        if self.force_no_layered_context:residual=torch.zeros_like(residual)
        output=super().forward(memory,valid,context+residual,**kwargs)
        if self.direct_sequence:
            correction=2.5*self.layered_sequence(torch.cat((context,observation*gate[:,:1],events*gate[:,1:]),-1)).tanh()
            if self.force_no_layered_context:correction=torch.zeros_like(correction)
            command=(output['commanded_acceleration_mps2']+correction).clamp(-8.,3.)
            speed,acceleration=integrate_acceleration(command,memory[:,-1,40]*40,memory[:,-1,42]*(100/3.6))
            output.update(commanded_acceleration_mps2=command,speed_sequence_mps=speed,
                acceleration_sequence_mps2=acceleration,target_speed_mps=speed[:,0],
                action_logits=F.one_hot(first_action(speed[:,0],acceleration[:,0]),len(ACTION_LABELS)).to(speed)*12,
                layered_sequence_correction=correction)
        output['layered_gates']=gate
        return output
