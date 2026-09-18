"""Online adapter for the existing multi-step executor; no first-step flattening."""

from copy import deepcopy
import json

from scene_understanding.src.control_plan_executor import advance_control_plan


class DrivingPlanRuntime:
    def __init__(self):
        self.document=None
        self.signature=None
        self.state=None
        self.feedback=None
        self.settled_since=None
        self.frame_id=None

    def enforce_execution(self,decision,canonical):
        """A later liveness or sequence controller cannot bypass plan readiness."""
        if canonical.get('decision_status')=='READY':return deepcopy(decision),False
        result=deepcopy(decision)
        emergency=bool(result.get('emergency')) or result.get('action')=='emergency_brake'
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
        self.frame_id=frame_id;self.feedback=None
        steps=self.document['intent']['steps']
        if not steps:raise ValueError('Empty executable plan')
        index=0 if self.state is None else self.state.get('active_step_index')
        if index is None:return None
        step=steps[index]
        kind=(step.get('completion') or {}).get('type')
        parameters=step.get('parameters') or {}
        action=step['action']
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
            satisfied=target is not None and abs(speed_mps-float(target))<=1./3.6
        elif kind=='LANE_CHANGE_COMPLETED':
            evidence=execution_state or {}
            satisfied=evidence.get('source_step_id')==step['step_id'] and evidence.get('pid',{}).get('lane_change_completed') is True
        if self.state is None or self.state.get('plan_status')!='ACTIVE' or self.state['step_states'][index]['status']!='ACTIVE':satisfied=False
        if satisfied:
            if self.settled_since is None:self.settled_since=timestamp_s
        else:self.settled_since=None
        required_hold=0. if kind=='TARGET_SPEED_REACHED' else .5
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
        state,decision=advance_control_plan(self.document,world,alignment,risk,
            prior_state=self.state,feedback=self.feedback)
        self.feedback=None
        index=state.get('active_step_index')
        if index is not None:
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
                decision.update(decision_status='BLOCKED',action='stop',target_speed_kmh=0.,
                    target_lane=None,target_location=None,emergency=False,
                    blocked_reason_codes=['await_observed_step_condition'],reason='await_observed_step_condition')
            elif state['step_states'][index]['status']=='WAITING':
                state['step_states'][index]['status']='ACTIVE'
        elif state['plan_status']=='COMPLETED' and self.document['intent']['steps'][-1]['action'] in ('STOP','EMERGENCY_BRAKE'):
            emergency=self.document['intent']['steps'][-1]['action']=='EMERGENCY_BRAKE'
            decision.update(action='emergency_brake' if emergency else 'stop',target_speed_kmh=0.,emergency=emergency)
        self.state=state
        return decision
