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
        self.effective_document=None
        self.last_execution_state={}
        self.settled_target_speed_mps=None
        self.speed_target_status='UNKNOWN'
        self.speed_constraint_codes=[]
        self.plan_origin_lane=None
        self.pending_lane_change=None
        # Deliberately survives request changes: a return instruction may be
        # dispatched as the command after the lane change it is undoing.
        self.lane_change_history=[]
        self.return_resolution=None

    @staticmethod
    def _lane_ref(value):
        """Normalize controller lane evidence without guessing topology."""
        if value is None:
            return None
        if isinstance(value, dict):
            lane_id=value.get('lane_id')
            if lane_id is None:return None
            return dict(road_id=value.get('road_id'),section_id=value.get('section_id'),lane_id=lane_id)
        if isinstance(value,(str,int)) and not isinstance(value,bool):
            return dict(road_id=None,section_id=None,lane_id=value)
        return None

    @classmethod
    def _pid_lane_ref(cls,pid,name):
        value=pid.get(name+'_lane_ref')
        if value is None:value=pid.get(name+'_lane')
        if value is None:value=pid.get(name+'_lane_id')
        if name=='current' and value is None:value=pid.get('current_lane_id')
        return cls._lane_ref(value)

    @staticmethod
    def _same_lane(first,second):
        if not first or not second or first.get('lane_id')!=second.get('lane_id'):
            return False
        for key in ('road_id','section_id'):
            if first.get(key) is not None and second.get(key) is not None and first[key]!=second[key]:
                return False
        return True

    @staticmethod
    def _finite_location(value):
        if not isinstance(value,dict) or not all(key in value for key in ('x','y')):
            return None
        keys=('x','y','z','yaw')
        result={}
        for key in keys:
            if key not in value:continue
            item=value[key]
            if isinstance(item,bool) or not isinstance(item,(int,float)) or not math.isfinite(float(item)):
                return None
            result[key]=float(item)
        result.setdefault('z',0.)
        reference=value.get('reference')
        if reference is not None:
            parsed=DrivingPlanRuntime._finite_location(reference)
            if parsed is None or 'yaw' not in parsed:return None
            result['reference']=parsed
        return result

    def _fresh_step_evidence(self,evidence,step,frame_id):
        return bool(isinstance(evidence,dict)
            and (evidence.get('frame_id') or evidence.get('observation_frame_id'))==frame_id
            and evidence.get('request_id')==self.document['request_id']
            and (evidence.get('step_id') or evidence.get('source_step_id'))==step['step_id'])

    def _road_target(self,step,frame_id,alignment,readiness):
        external=(readiness or {}).get(step['step_id']) or {}
        if (self._fresh_step_evidence(external,step,frame_id)
                and external.get('reachable') is True
                and external.get('valid',True) is True
                and external.get('ambiguous',False) is False
                and isinstance(external.get('source'),str)
                and bool(external.get('source'))):
            target=self._finite_location(external.get('target_location') or external.get('road_target'))
            if target is not None:return target
        if not isinstance(alignment,dict) or alignment.get('world_state_frame_id')!=frame_id:
            return None
        item=next((value for value in alignment.get('step_alignments',[])
            if value.get('step_id')==step['step_id']),{})
        if item.get('alignment_success') is not True:return None
        matched=item.get('matched_entity') or {}
        return self._finite_location(item.get('target_location')
            or matched.get('target_location') or matched.get('location'))

    def _resolve_return(self,step,evidence):
        pid=(evidence or {}).get('pid') or {}
        current=self._pid_lane_ref(pid,'current')
        event=next((item for item in reversed(self.lane_change_history)
            if not item.get('returned')),None)
        original=(event or {}).get('source') or self.plan_origin_lane
        result=dict(status='UNKNOWN',direction=None,original_lane_ref=deepcopy(original),
            current_lane_ref=deepcopy(current),reason='original_lane_reference_missing')
        if original is None or current is None:return result
        if self._same_lane(current,original):
            result.update(status='ALREADY_IN_ORIGINAL_LANE',reason='already_in_original_lane')
            return result
        for direction in ('left','right'):
            adjacent=self._pid_lane_ref(pid,direction)
            if self._same_lane(adjacent,original):
                if pid.get(direction+'_lane_change_allowed') is False:
                    result.update(status='UNREACHABLE',
                        reason='original_lane_change_not_legal')
                    return result
                result.update(status='DIRECTION_RESOLVED',direction=direction,
                    reason='original_lane_adjacent_'+direction)
                return result
        result.update(status='UNREACHABLE',reason='original_lane_not_adjacent')
        return result

    def _effective_step(self,step,frame_id,evidence,alignment,readiness):
        result=deepcopy(step)
        parameters=result.setdefault('parameters',{})
        action=str(result.get('action','')).upper()
        if action=='TURN' and parameters.get('target_location') is None:
            target=self._road_target(result,frame_id,alignment,readiness)
            if target is not None:parameters['target_location']=target
        if action=='CHANGE_LANE' and parameters.get('return_to')=='ORIGINAL_LANE':
            self.return_resolution=self._resolve_return(result,evidence)
            direction=self.return_resolution.get('direction')
            if direction in ('left','right'):
                parameters['direction']=direction.upper()
                parameters['original_lane_ref']=deepcopy(
                    self.return_resolution.get('original_lane_ref')
                )
            else:parameters.pop('direction',None)
        else:self.return_resolution=None
        return result

    def _record_lane_change(self,step,evidence,completed):
        parameters=step.get('parameters') or {}
        if str(step.get('action','')).upper()!='CHANGE_LANE':return
        if parameters.get('return_to')=='ORIGINAL_LANE':
            if completed:
                event=next((item for item in reversed(self.lane_change_history)
                    if not item.get('returned')),None)
                if event is not None:event['returned']=True
            return
        direction=str(parameters.get('direction','')).lower()
        if direction not in ('left','right'):return
        pid=(evidence or {}).get('pid') or {}
        current=self._pid_lane_ref(pid,'current')
        identity=(self.document['request_id'],step['step_id'])
        if self.pending_lane_change is None or self.pending_lane_change.get('identity')!=identity:
            self.pending_lane_change=dict(identity=identity,source=deepcopy(current),direction=direction)
        if not completed:return
        source=self.pending_lane_change.get('source')
        if source is not None and current is not None and not self._same_lane(source,current):
            self.lane_change_history.append(dict(source=deepcopy(source),target=deepcopy(current),
                direction=direction,request_id=identity[0],step_id=identity[1],returned=False))
        self.pending_lane_change=None

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

    def prepare(self,document,*,frame_id,timestamp_s,speed_mps,execution_state=None,
                alignment=None,readiness=None):
        signature=json.dumps(document,sort_keys=True,ensure_ascii=False)
        if signature!=self.signature:
            self.document=deepcopy(document);self.signature=signature
            self.state=None;self.settled_since=None
            self.risk_pause=False;self.risk_clear_since=None;self.risk_clear_frame=None
            self.risk_clear_last_time=None
            self.completion_step=None
            self.condition_key=None;self.condition_since=None
            self.condition_time=None;self.condition_frame=None
            self.settled_target_speed_mps=None
            self.speed_target_status='UNKNOWN';self.speed_constraint_codes=[]
            self.pending_lane_change=None;self.return_resolution=None
            self.plan_origin_lane=self._pid_lane_ref((execution_state or {}).get('pid') or {},'current')
        self.frame_id=frame_id;self.feedback=None
        self.timestamp_s=timestamp_s
        self.last_execution_state=deepcopy(execution_state or {})
        steps=self.document['intent']['steps']
        if not steps:raise ValueError('Empty executable plan')
        index=0 if self.state is None else self.state.get('active_step_index')
        if index is None:return None
        step=self._effective_step(steps[index],frame_id,execution_state or {},alignment,readiness)
        self.effective_document=deepcopy(self.document)
        self.effective_document['intent']['steps'][index]=deepcopy(step)
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
            pid=evidence.get('pid') or {}
            status=evidence.get('speed_target_status') or pid.get('speed_target_status') or 'UNKNOWN'
            effective=evidence.get('effective_target_speed_kmh')
            if effective is None:effective=pid.get('effective_target_speed_kmh')
            codes=evidence.get('speed_constraint_codes')
            if codes is None:codes=pid.get('speed_constraint_codes') or []
            self.speed_target_status=str(status).upper()
            self.speed_constraint_codes=[str(value) for value in codes if str(value)]
            active_speed_step=(self.state is not None
                and self.state.get('plan_status')=='ACTIVE'
                and self.state.get('active_step_id')==step['step_id']
                and self.state['step_states'][index]['status']=='ACTIVE')
            if self.speed_target_status=='UNREACHABLE' and fresh and active_speed_step:
                self.feedback=dict(schema_version='1.0.0',request_id=document['request_id'],
                    frame_id=frame_id,step_id=step['step_id'],outcome='FAILED',
                    reason_codes=['target_speed_unreachable']+self.speed_constraint_codes)
                self.settled_since=None
            elif (self.speed_target_status=='CONSTRAINED' and fresh
                    and isinstance(effective,(int,float)) and not isinstance(effective,bool)
                    and math.isfinite(float(effective)) and float(effective)>=0.):
                constrained=float(effective)/3.6
                if target is None or constrained<float(target):target=constrained
            if target is not None:
                if (self.settled_target_speed_mps is None
                        or abs(self.settled_target_speed_mps-float(target))>1./3.6):
                    self.settled_since=None
                self.settled_target_speed_mps=float(target)
            satisfied=(self.feedback is None and target is not None
                and abs(speed_mps-float(target))<=1./3.6)
        elif kind=='LANE_CHANGE_COMPLETED':
            evidence=execution_state or {}
            pid=evidence.get('pid',{})
            if self.return_resolution and self.return_resolution.get('status')=='ALREADY_IN_ORIGINAL_LANE':
                lateral=pid.get('lateral_error_m');heading=pid.get('heading_error_deg')
                satisfied=(fresh and isinstance(lateral,(int,float)) and isinstance(heading,(int,float))
                    and math.isfinite(lateral) and math.isfinite(heading)
                    and abs(lateral)<=.35 and abs(heading)<=5.)
            else:
                satisfied=fresh and pid.get('lane_change_completed') is True
            self._record_lane_change(step,evidence,False)
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
        if self.feedback is not None and self.feedback.get('outcome')=='FAILED':
            satisfied=False
        elif satisfied:
            if self.settled_since is None:self.settled_since=timestamp_s
        else:self.settled_since=None
        # A single matching sample is not physical completion.  The same
        # half-second hold is required after relative-target resolution so
        # braking/acceleration overshoot cannot advance the next plan step.
        required_hold=.5
        if self.settled_since is not None and timestamp_s-self.settled_since>=required_hold:
            reasons=['observed_physical_completion']
            if kind=='TARGET_SPEED_REACHED' and self.speed_target_status=='CONSTRAINED':
                reasons=['observed_constrained_speed_completion']+self.speed_constraint_codes
            elif self.return_resolution and self.return_resolution.get('status')=='ALREADY_IN_ORIGINAL_LANE':
                reasons=['already_in_original_lane']
            self.feedback=dict(schema_version='1.0.0',request_id=document['request_id'],
                frame_id=frame_id,step_id=step['step_id'],outcome='COMPLETED',
                reason_codes=list(dict.fromkeys(reasons)))
            if kind=='LANE_CHANGE_COMPLETED':
                self._record_lane_change(step,evidence,True)
            self.settled_since=None
            index+=1
        if index>=len(steps):return None
        if index!=(0 if self.state is None else self.state.get('active_step_index')):
            next_step=self._effective_step(steps[index],frame_id,execution_state or {},alignment,readiness)
            self.effective_document['intent']['steps'][index]=deepcopy(next_step)
            return next_step
        return step

    def advance(self,world,risk,*,alignment=None,readiness=None):
        risk=deepcopy(risk)
        if risk.get('recommended_action')=='keep_lane':risk['recommended_action']='maintain_speed'
        risk.setdefault('reason_codes',[])
        for direction in ('left','right'):
            risk.setdefault('lane_change',{}).setdefault(direction,dict(is_safe=False,reason_codes=['lane_observation_missing']))
        if alignment is None:
            aligned=[]
            for step in (self.effective_document or self.document)['intent']['steps']:
                params=step.get('parameters') or {}
                required=bool(step.get('target') or step.get('target_ref') or params.get('target') or params.get('target_ref'))
                aligned.append(dict(step_id=step['step_id'],alignment_required=required,
                    alignment_success=not required,reason_code='target_observation_missing' if required else 'not_required',matched_entity=None))
            alignment=dict(request_id=self.document['request_id'],world_state_frame_id=world['frame_id'],
                parse_status=self.document['parse_result']['status'],step_alignments=aligned)
        if risk.get('risk_level')!='low':
            self.feedback=None;self.settled_since=None
        self._observe_condition_completion(world,risk,readiness)
        effective=self.effective_document or self.document
        state,decision=advance_control_plan(effective,world,alignment,risk,
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
            step=effective['intent']['steps'][index]
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
            parameters=step.get('parameters') or {}
            if (step.get('action')=='CHANGE_LANE'
                    and parameters.get('return_to')=='ORIGINAL_LANE'
                    and (self.return_resolution or {}).get('status') in ('UNKNOWN','UNREACHABLE')):
                reason=(self.return_resolution or {}).get('reason','original_lane_reference_missing')
                state['step_states'][index]['status']='WAITING'
                state['step_states'][index]['reason_codes']=[reason]
                state['reason_codes']=[reason]
                decision.update(decision_status='BLOCKED',action='stop',target_speed_kmh=0.,
                    target_lane=None,target_location=None,emergency=False,reason=reason,
                    blocked_reason_codes=[reason])
            if (step.get('action') in ('SET_SPEED','ADJUST_SPEED')
                    and self.speed_target_status=='CONSTRAINED'):
                reasons=['target_speed_constrained']+self.speed_constraint_codes
                state['step_states'][index]['reason_codes']=list(dict.fromkeys(reasons))
                state['reason_codes']=list(dict.fromkeys(reasons))
        elif state['plan_status']=='COMPLETED' and self.document['intent']['steps'][-1]['action'] in ('STOP','EMERGENCY_BRAKE'):
            emergency=self.document['intent']['steps'][-1]['action']=='EMERGENCY_BRAKE'
            decision.update(action='emergency_brake' if emergency else 'stop',target_speed_kmh=0.,emergency=emergency)
        self.state=state
        return decision

    def _observe_condition_completion(self,world,risk,readiness=None):
        """Only explicit, freshly observed predicates complete a check/wait."""
        index=None if self.state is None else self.state.get('active_step_index')
        effective=self.effective_document or self.document
        step=effective['intent']['steps'][index] if index is not None else None
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
            if eligible:
                observed=(readiness or {}).get(step['step_id']) or {}
                status=str(observed.get('status','')).upper()
                observed_at=observed.get('observed_at_s')
                valid_until=observed.get('valid_until_s')
                time_valid=(observed_at is None or (
                    isinstance(observed_at,(int,float)) and not isinstance(observed_at,bool)
                    and math.isfinite(float(observed_at))
                    and 0.<=self.timestamp_s-float(observed_at)<=.25))
                expiry_valid=(valid_until is None or (
                    isinstance(valid_until,(int,float)) and not isinstance(valid_until,bool)
                    and math.isfinite(float(valid_until)) and self.timestamp_s<=float(valid_until)))
                explicit=(observed.get('satisfied') is True
                    or status in ('SATISFIED','CLEAR','COMPLETED'))
                clear=(fresh and risk.get('risk_level')=='low' and not self.risk_pause
                    and self._fresh_step_evidence(observed,step,world['frame_id'])
                    and observed.get('condition')==condition
                    and explicit and observed.get('valid',True) is True
                    and isinstance(observed.get('source'),str) and bool(observed.get('source'))
                    and time_valid and expiry_valid)
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
