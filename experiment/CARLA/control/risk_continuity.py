"""Bounded visual-risk hysteresis gated by fresh, changing physical evidence.

Only global visual risk is smoothed. Lane permissions, emergency responses,
physical caution, and traffic-control contracts remain authoritative.
"""
from dataclasses import dataclass
import math

LEVELS=('low','medium','high')


@dataclass(frozen=True)
class RiskContinuityConfig:
    smoothing_tau_s: float=.25
    medium_confirm_s: float=.35
    release_confirm_s: float=.6
    maximum_gap_s: float=.3
    sudden_gap_drop_m: float=3.
    relative_speed_change_mps: float=1.5
    closing_emergency_ttc_s: float=3.

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value=getattr(self,name)
            if isinstance(value,bool) or not math.isfinite(value) or value<=0:
                raise ValueError(f'{name} must be finite and positive')


def number(value):
    return float(value) if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) else None


class RiskContinuityFilter:
    def __init__(self,config=None):
        self.config=config or RiskContinuityConfig()
        self.reset()

    def reset(self):
        self.time=None;self.frame=None;self.level=None;self.ema=None
        self.snapshot=None;self.pending=None;self.pending_since=None

    def _snapshot(self,risk,tracked):
        radar=risk.get('physical_forward_radar') or {}
        nearest=(tracked or {}).get('nearest_obstacle') or {}
        distance=number(radar.get('nearest_distance_m'))
        relative=number(radar.get('nearest_relative_velocity_mps'))
        return dict(present=distance is not None, distance=distance,relative=relative,
            nearest_track_id=nearest.get('track_id'),
            selected_track_id=(tracked or {}).get('selected_track_id'))

    def apply(self,risk,*,timestamp_s,frame_id,tracked_observation=None):
        result=dict(risk)
        raw=str(risk.get('risk_level','unknown')).lower()
        diagnostic=dict(schema_version='risk_continuity/1.0',raw_level=raw,
            effective_level=raw,changed=False,external_event=False,reason='raw_passthrough')
        now=number(timestamp_s);evidence=risk.get('risk_evidence') or {}
        radar=risk.get('physical_forward_radar') or {}
        snapshot=self._snapshot(risk,tracked_observation)
        dt=now-self.time if now is not None and self.time is not None else None
        events=[]
        if self.snapshot is not None and dt is not None and 0<dt<=self.config.maximum_gap_s:
            if snapshot['present']!=self.snapshot['present']:events.append('front_presence_changed')
            for key in ('nearest_track_id','selected_track_id'):
                if snapshot[key]!=self.snapshot[key]:events.append(key+'_changed')
            old=self.snapshot['distance'];new=snapshot['distance']
            if old is not None and new is not None:
                predicted=old+(self.snapshot['relative'] or 0.)*dt
                if predicted-new>=self.config.sudden_gap_drop_m:events.append('unexpected_gap_drop')
            old=self.snapshot['relative'];new=snapshot['relative']
            if old is not None and new is not None and old-new>=self.config.relative_speed_change_mps:
                events.append('front_deceleration_or_closing_change')
        closing=number(radar.get('nearest_closing_velocity_mps'))
        closing_distance=number(radar.get('nearest_closing_distance_m'))
        fresh=(evidence.get('is_fresh') is True and number(evidence.get('age_s')) is not None
               and 0<=evidence['age_s']<=.25)
        ttc=(closing_distance/closing if closing is not None and closing>0 and closing_distance is not None and closing_distance>0 else None)
        if fresh and ttc is not None and ttc<=self.config.closing_emergency_ttc_s:
            result.update(risk_level='high',recommended_action='emergency_brake',
                reason_codes=list(dict.fromkeys(list(result.get('reason_codes') or [])+['temporal_physical_closing_ttc'])))
            raw='high';diagnostic.update(effective_level='high',changed=risk.get('risk_level')!='high')
            events.append('physical_closing_ttc')
        if risk.get('recommended_action')=='emergency_brake' and raw!='high':
            raw='high';result['risk_level']='high'
        physical=any(str(c).startswith('physical_forward_radar_') for c in risk.get('reason_codes',[]))
        hard=(raw=='high' or risk.get('recommended_action')=='emergency_brake' or physical or 'physical_closing_ttc' in events)
        probabilities=risk.get('learned_probabilities') or risk.get('probabilities') or {}
        values=[number(probabilities.get(k)) for k in LEVELS]
        valid=(all(v is not None and 0<=v<=1 for v in values) and abs(sum(values)-1.)<=.02)
        continuity=(fresh and now is not None and raw in LEVELS and valid)
        if not continuity or self.time is None or dt is None or dt<0 or dt>self.config.maximum_gap_s:
            self.reset()
            if continuity:
                self.level=raw;self.ema=values;self.time=now;self.frame=frame_id;self.snapshot=snapshot
            diagnostic.update(reason='cold_start' if continuity else 'freshness_or_probability_unavailable',
                              event_reasons=events,external_event=bool(events))
            return result,diagnostic
        diagnostic.update(event_reasons=events,external_event=bool(events),closing_ttc_s=ttc)
        if hard:
            self.level=raw;self.ema=[float(k==raw) for k in LEVELS]
            self.pending=None;self.pending_since=None
            diagnostic['reason']='immediate_safety_evidence'
        elif events:
            # New evidence may escalate immediately, never immediately erase
            # a more conservative established state.
            self.ema=values
            if LEVELS.index(raw)>=LEVELS.index(self.level):
                self.level=raw;self.pending=None;self.pending_since=None
            diagnostic['reason']='external_event_reassessment'
        elif dt>0 and frame_id!=self.frame:
            alpha=1.-math.exp(-dt/self.config.smoothing_tau_s)
            self.ema=[a+alpha*(b-a) for a,b in zip(self.ema,values)]
        candidate=LEVELS[max(range(3),key=lambda i:self.ema[i])]
        if candidate=='high':candidate='medium'  # Only current high/emergency evidence raises HIGH.
        if not hard:
            if candidate==self.level:
                self.pending=None;self.pending_since=None
            elif frame_id!=self.frame and dt>0:
                if candidate!=self.pending:
                    self.pending=candidate;self.pending_since=now
                hold=self.config.medium_confirm_s if LEVELS.index(candidate)>LEVELS.index(self.level) else self.config.release_confirm_s
                if now-self.pending_since>=hold:
                    self.level=candidate;self.pending=None;self.pending_since=None
            if self.level!=raw:
                result.update(risk_level=self.level,
                    recommended_action={'low':'keep_lane','medium':'decelerate','high':'emergency_brake'}[self.level],
                    probabilities=dict(zip(LEVELS,self.ema)),
                    reason_codes=[c for c in result.get('reason_codes',[]) if not str(c).startswith('learned_visual_risk_')])
                if self.level!='low':result['reason_codes'].append('temporal_visual_risk_'+self.level)
                result['source']=str(result.get('source','visual_risk'))+'+risk_continuity'
                diagnostic['reason']='stable_event_hysteresis'
        self.time=now;self.frame=frame_id;self.snapshot=snapshot
        diagnostic.update(effective_level=result['risk_level'],changed=result!=risk,
                          smoothed_probabilities=dict(zip(LEVELS,self.ema)),
                          pending_level=self.pending,pending_since_s=self.pending_since)
        # Raw classifications/probabilities remain separately inspectable.
        return result,diagnostic
