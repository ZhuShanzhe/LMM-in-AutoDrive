"""Online adapter for the existing multi-step executor; no first-step flattening."""

from copy import deepcopy
import json
import math

from scene_understanding.src.control_plan_executor import advance_control_plan


class DrivingPlanRuntime:
    def __init__(self):
        self.document=None
        self.signature=None
        self.state=None
        self.feedback=None
        self.settled_since=None
        self.frame_id=None
        self.timestamp_s=None
        self.risk_pause=False
        self.risk_clear_since=None
        self.risk_clear_frame=None
        self.risk_clear_last_time=None
        self.completion_step=None
        self.completion_time=None
        self.completion_frame=None
        self.completion_lane=None
        self.junction_entered=False
        self.junction_heading=None
        self.condition_key=None
        self.condition_since=None
        self.condition_time=None
        self.condition_frame=None

    def enforce_execution(self,decision,canonical):
        """A later liveness or sequence controller cannot bypass plan readiness."""
        if canonical.get('decision_status')=='READY':return deepcopy(decision),False
        result=deepcopy(decision)
        emergency=(bool(result.get('emergency')) or result.get('action')=='emergency_brake'
            or bool(canonical.get('emergency')) or canonical.get('action')=='emergency_brake')
        result.update(action='emergency_brake' if emergency else 'stop',target_speed_kmh=0.,
            target_lane=None,target_location=None,emergency=emergency,
            allow_positive_acceleration=False,decision_status=canonical['decision_status'],
            reason='active_plan_not_ready',blocked_reason_codes=canonical.get('blocked_reason_codes',[]))
        for key in ('longitudinal_sequence_schema','target_acceleration_mps2','sequence_valid_until_s'):
            result.pop(key,None)
        return result,result!=decision

    def prepare(self,document,*,frame_id,timestamp_s,speed_mps,execution_state=None):
        signature=json.dumps(document,sort_keys=True,ensure_ascii=False)
        if signature!=self.signature:
            self.document=deepcopy(document);self.signature=signature
            self.state=None;self.settled_since=None
            self.risk_pause=False;self.risk_clear_since=None;self.risk_clear_frame=None
            self.risk_clear_last_time=None
            self.completion_step=None
            self.condition_key=None;self.condition_since=None
            self.condition_time=None;self.condition_frame=None
        self.frame_id=frame_id;self.feedback=None
        self.timestamp_s=timestamp_s
        steps=self.document['intent']['steps']
        if not steps:raise ValueError('Empty executable plan')
        index=0 if self.state is None else self.state.get('active_step_index')
        if index is None:return None
        step=steps[index]
        kind=(step.get('completion') or {}).get('type')
        parameters=step.get('parameters') or {}
        action=step['action']
        evidence=execution_state or {}
        fresh=(evidence.get('observation_frame_id')==frame_id
            and evidence.get('source_step_id')==step['step_id'])
        continuous=(self.completion_time is not None and 0 < timestamp_s-self.completion_time <= .25
            and frame_id!=self.completion_frame)
        if self.completion_step!=step['step_id']:
            self.completion_step=step['step_id'];self.completion_lane=None
            self.junction_entered=False;self.junction_heading=None;self.settled_since=None
        if not continuous:self.settled_since=None
        self.completion_time=timestamp_s;self.completion_frame=frame_id
        if kind is None and index+1<len(steps):
            kind={'STOP':'VEHICLE_STOPPED','EMERGENCY_BRAKE':'VEHICLE_STOPPED',
                'ADJUST_SPEED':'TARGET_SPEED_REACHED','SET_SPEED':'TARGET_SPEED_REACHED',
                'CHANGE_LANE':'LANE_CHANGE_COMPLETED'}.get(action)
        satisfied=False
        if kind=='VEHICLE_STOPPED':satisfied=speed_mps<.25
        elif kind=='TARGET_SPEED_REACHED':
            target=parameters.get('target_speed_mps')
            if target is None and parameters.get('target_speed_kmh') is not None:
                target=parameters['target_speed_kmh']/3.6
            if target is None and self.state is not None:
                step_state=next((item for item in self.state.get('step_states',[])
                    if item.get('step_id')==step['step_id']),{})
                resolved=step_state.get('resolved_target_speed_kmh')
                if resolved is not None:target=float(resolved)/3.6
            satisfied=target is not None and abs(speed_mps-float(target))<=1./3.6
        elif kind=='LANE_CHANGE_COMPLETED':
            evidence=execution_state or {}
            satisfied=evidence.get('source_step_id')==step['step_id'] and evidence.get('pid',{}).get('lane_change_completed') is True
        elif kind in ('ACTION_REACHED','JUNCTION_EXITED') and fresh:
            pid=evidence.get('pid') or {}
            lateral=pid.get('lateral_error_m');heading=pid.get('heading_error_deg')
            centered=(isinstance(lateral,(int,float)) and isinstance(heading,(int,float))
                and math.isfinite(lateral) and math.isfinite(heading)
                and abs(lateral)<=.35 and abs(heading)<=5.
                and not pid.get('emergency_latched',False))
            lane=pid.get('current_lane_id')
            if action=='KEEP_LANE' and not parameters:
                if self.completion_lane is None:self.completion_lane=lane
                satisfied=(centered and lane is not None and lane==self.completion_lane
                    and pid.get('in_junction') is False and speed_mps>.5)
                if lane!=self.completion_lane:self.completion_lane=lane;self.settled_since=None
            elif ((action=='PROCEED' and parameters.get('condition')=='STRAIGHT_THROUGH_JUNCTION')
                    or (kind=='JUNCTION_EXITED' and action in ('TURN','U_TURN'))):
                yaw=evidence.get('ego_heading_deg')
                active=(self.state is not None and self.state.get('plan_status')=='ACTIVE'
                    and self.state['step_states'][index]['status']=='ACTIVE' and not self.risk_pause)
                if active and pid.get('in_junction') is True and not self.junction_entered:
                    self.junction_entered=True;self.junction_heading=yaw
                heading_valid=(isinstance(yaw,(int,float)) and isinstance(self.junction_heading,(int,float))
                    and math.isfinite(yaw) and math.isfinite(self.junction_heading)
                    )
                delta=(yaw-self.junction_heading+180)%360-180 if heading_valid else None
                direction=str(parameters.get('direction','')).upper()
                aligned=(heading_valid and (
                    (action=='PROCEED' and abs(delta)<=20.)
                    or (action=='TURN' and direction=='RIGHT' and 30.<=delta<=150.)
                    or (action=='TURN' and direction=='LEFT' and -150.<=delta<=-30.)
                    or (action=='U_TURN' and abs(delta)>=150.)))
                satisfied=(self.junction_entered and pid.get('in_junction') is False
                    and centered and aligned and speed_mps>.5)
        if self.state is None or self.state.get('plan_status')!='ACTIVE' or self.state['step_states'][index]['status']!='ACTIVE':satisfied=False
        if satisfied:
            if self.settled_since is None:self.settled_since=timestamp_s
        else:self.settled_since=None
        # A single matching sample is not physical completion.  The same
        # half-second hold is required after relative-target resolution so
        # braking/acceleration overshoot cannot advance the next plan step.
        required_hold=.5
        if self.settled_since is not None and timestamp_s-self.settled_since>=required_hold:
            self.feedback=dict(schema_version='1.0.0',request_id=document['request_id'],
                frame_id=frame_id,step_id=step['step_id'],outcome='COMPLETED',
                reason_codes=['observed_physical_completion'])
            self.settled_since=None
            index+=1
        return steps[index] if index<len(steps) else None

    def advance(self,world,risk,*,alignment=None,readiness=None):
        risk=deepcopy(risk)
        if risk.get('recommended_action')=='keep_lane':risk['recommended_action']='maintain_speed'
        risk.setdefault('reason_codes',[])
        for direction in ('left','right'):
            risk.setdefault('lane_change',{}).setdefault(direction,dict(is_safe=False,reason_codes=['lane_observation_missing']))
        if alignment is None:
            aligned=[]
            for step in self.document['intent']['steps']:
                params=step.get('parameters') or {}
                required=bool(step.get('target') or step.get('target_ref') or params.get('target') or params.get('target_ref'))
                aligned.append(dict(step_id=step['step_id'],alignment_required=required,
                    alignment_success=not required,reason_code='target_observation_missing' if required else 'not_required',matched_entity=None))
            alignment=dict(request_id=self.document['request_id'],world_state_frame_id=world['frame_id'],
                parse_status=self.document['parse_result']['status'],step_alignments=aligned)
        if risk.get('risk_level')!='low':
            self.feedback=None;self.settled_since=None
        self._observe_condition_completion(world,risk)
        state,decision=advance_control_plan(self.document,world,alignment,risk,
            prior_state=self.state,feedback=self.feedback,transient_risk_wait=True)
        self.feedback=None
        index=state.get('active_step_index')
        if index is not None:
            transient_reasons={'risk_requires_emergency_brake','risk_requires_deceleration'}
            if transient_reasons.intersection(decision.get('blocked_reason_codes',[])):
                self.risk_pause=True
            if self.risk_pause:
                clear=(risk.get('frame_id')==world['frame_id'] and risk.get('risk_level')=='low'
                    and risk.get('recommended_action')=='maintain_speed'
                    and decision.get('decision_status')=='READY')
                if not clear:
                    self.risk_clear_since=None;self.risk_clear_frame=None
                    self.risk_clear_last_time=None
                elif (self.risk_clear_since is None or self.risk_clear_last_time is None
                        or not 0. <= self.timestamp_s-self.risk_clear_last_time <= .5):
                    self.risk_clear_since=self.timestamp_s;self.risk_clear_frame=world['frame_id']
                    self.risk_clear_last_time=self.timestamp_s
                elif (world['frame_id']!=self.risk_clear_frame and
                        self.timestamp_s-self.risk_clear_since>=.5):
                    self.risk_pause=False;self.risk_clear_since=None;self.risk_clear_frame=None
                    self.risk_clear_last_time=None
                elif world['frame_id']!=self.risk_clear_frame:
                    self.risk_clear_frame=world['frame_id'];self.risk_clear_last_time=self.timestamp_s
                if self.risk_pause:
                    state['step_states'][index]['status']='WAITING'
                    state['reason_codes']=['await_sustained_risk_clearance']
                    emergency=decision.get('action')=='emergency_brake' or bool(decision.get('emergency'))
                    decision.update(decision_status='BLOCKED',action='emergency_brake' if emergency else 'stop',
                        target_speed_kmh=0.,target_lane=None,target_location=None,emergency=emergency,
                        reason='await_sustained_risk_clearance',blocked_reason_codes=['await_sustained_risk_clearance'])
            step=self.document['intent']['steps'][index]
            trigger=step.get('trigger') or {'type':'IMMEDIATE'}
            completed={item['step_id'] for item in state['step_states'] if item['status']=='COMPLETED'}
            trigger_ready=trigger.get('type')=='IMMEDIATE' or (
                trigger.get('type')=='AFTER_STEP' and trigger.get('step_id') in completed)
            external=(readiness or {}).get(step['step_id']) or {}
            external_ready=external.get('frame_id')==world['frame_id'] and external.get('satisfied') is True
            if trigger.get('type') not in ('IMMEDIATE','AFTER_STEP'):
                trigger_ready=external_ready
            for condition in step.get('preconditions') or []:
                if condition=='PATH_CLEAR':
                    satisfied=risk.get('frame_id')==world['frame_id'] and risk.get('risk_level')=='low'
                elif condition=='TARGET_LANE_SAFE':
                    direction=str((step.get('parameters') or {}).get('direction','')).lower()
                    satisfied=risk.get('lane_change',{}).get(direction,{}).get('is_safe') is True
                elif condition=='TARGET_VISIBLE':
                    aligned=next((x for x in alignment.get('step_alignments',[]) if x.get('step_id')==step['step_id']),{})
                    satisfied=bool(aligned.get('alignment_success') and aligned.get('matched_entity'))
                else:satisfied=external_ready
                trigger_ready=trigger_ready and satisfied
            if not trigger_ready:
                state['step_states'][index]['status']='WAITING'
                state['reason_codes']=['await_observed_step_condition']
                emergency=decision.get('action')=='emergency_brake' or bool(decision.get('emergency'))
                decision.update(decision_status='BLOCKED',action='emergency_brake' if emergency else 'stop',target_speed_kmh=0.,
                    target_lane=None,target_location=None,emergency=emergency,
                    blocked_reason_codes=['await_observed_step_condition'],reason='await_observed_step_condition')
            elif state['step_states'][index]['status']=='WAITING' and decision.get('decision_status')=='READY':
                state['step_states'][index]['status']='ACTIVE'
        elif state['plan_status']=='COMPLETED' and self.document['intent']['steps'][-1]['action'] in ('STOP','EMERGENCY_BRAKE'):
            emergency=self.document['intent']['steps'][-1]['action']=='EMERGENCY_BRAKE'
            decision.update(action='emergency_brake' if emergency else 'stop',target_speed_kmh=0.,emergency=emergency)
        self.state=state
        return decision

    def _observe_condition_completion(self,world,risk):
        """Only explicit, freshly observed predicates complete a check/wait."""
        index=None if self.state is None else self.state.get('active_step_index')
        step=self.document['intent']['steps'][index] if index is not None else None
        condition=(step.get('parameters') or {}).get('condition') if step else None
        eligible=(step is not None and self.state.get('plan_status')=='ACTIVE'
                  and step['action'] in ('CHECK','WAIT','CONFIRM'))
        fresh=(risk.get('frame_id')==world['frame_id']==self.frame_id)
        clear=fresh and risk.get('risk_level')=='low' and not self.risk_pause
        if condition in ('LEFT_LANE_SAFE','RIGHT_LANE_SAFE'):
            direction=condition.split('_')[0].lower()
            clear=clear and risk.get('lane_change',{}).get(direction,{}).get('is_safe') is True
        elif condition!='PATH_CLEAR':
            clear=False
        key=(self.document['request_id'],step['step_id'],condition) if eligible else None
        continuous=(self.condition_time is not None and self.timestamp_s is not None
                    and 0 < self.timestamp_s-self.condition_time<=.25
                    and self.condition_frame!=self.frame_id)
        if key!=self.condition_key or not continuous or not eligible or not clear:
            self.condition_since=self.timestamp_s if eligible and clear else None
        self.condition_key=key;self.condition_time=self.timestamp_s;self.condition_frame=self.frame_id
        if self.condition_since is not None and self.timestamp_s-self.condition_since>=.5:
            self.feedback=dict(schema_version='1.0.0',request_id=self.document['request_id'],
                frame_id=self.frame_id,step_id=step['step_id'],outcome='COMPLETED',
                reason_codes=['observed_condition_completed'])
            self.condition_since=None
