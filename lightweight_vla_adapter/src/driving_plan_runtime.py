"""Online adapter for the existing multi-step executor; no first-step flattening."""

from copy import deepcopy
import json
import math

from scene_understanding.src.control_plan_executor import advance_control_plan
from .risk_behavior import cautious_permission, FORWARD_STEPS
from .safety_adjusted_speed import safety_adjusted_speed_evidence


def qualitative_deceleration_reached(step, step_state, speed_kmh, tolerance_kmh):
    """A qualitative slowdown is an upper bound, not an exact numeric speed goal."""
    parameters=step.get('parameters') or {}
    if (step.get('action')!='ADJUST_SPEED' or parameters.get('change')!='DECREASE'
            or any(parameters.get(key) is not None for key in (
                'target_speed_mps','target_speed_kmh','speed_delta_mps','speed_delta_kmh'))):
        return False
    reference=step_state.get('speed_reference_kmh')
    target=step_state.get('resolved_target_speed_kmh')
    values=(reference,target,speed_kmh,tolerance_kmh)
    if any(isinstance(value,bool) or not isinstance(value,(int,float))
            or not math.isfinite(value) or value<0 for value in values):
        return False
    return target<reference-1e-6 and speed_kmh<=target+tolerance_kmh


class DrivingPlanRuntime:
    def __init__(self, *, speed_completion_hold_s=.5, speed_completion_tolerance_kmh=1.,
            safety_adjusted_completion=False):
        if not isinstance(safety_adjusted_completion,bool):
            raise ValueError('Safety adjusted completion must be boolean')
        self.safety_adjusted_completion=safety_adjusted_completion
        self.speed_adjustment_seen=None
        self.last_speed_completion_evidence=None
        self.previous_observed_speed=None
        self.previous_speed_time=None
        if (isinstance(speed_completion_hold_s,bool)
                or not isinstance(speed_completion_hold_s,(int,float))
                or not math.isfinite(speed_completion_hold_s)
                or not .5 <= speed_completion_hold_s <= 10.):
            raise ValueError('Speed completion hold must be between .5 and 10 seconds')
        self.speed_completion_hold_s=float(speed_completion_hold_s)
        if (isinstance(speed_completion_tolerance_kmh,bool)
                or not isinstance(speed_completion_tolerance_kmh,(int,float))
                or not math.isfinite(speed_completion_tolerance_kmh)
                or not .1<=speed_completion_tolerance_kmh<=3.):
            raise ValueError('Speed completion tolerance must be between .1 and 3 km/h')
        self.speed_completion_tolerance_kmh=float(speed_completion_tolerance_kmh)
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
        self.completed_maneuver_facts=[]
        self.maneuver_confirmation_bindings={}
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

    def _continue_return_settling(self,result,step,evidence):
        previous=self.return_resolution or {}
        pid=(evidence or {}).get('pid') or {}
        direction=previous.get('direction')
        lateral=pid.get('lateral_error_m');heading=pid.get('heading_error_deg')
        finite=all(isinstance(value,(int,float)) and not isinstance(value,bool)
            and math.isfinite(value) for value in (lateral,heading))
        command_id=(self.document or {}).get('request_id','')+':'+step['step_id']
        if (previous.get('status')=='DIRECTION_RESOLVED' and direction in ('left','right') and finite
                and (abs(lateral)>.35 or abs(heading)>5.)
                and (evidence or {}).get('observation_frame_id')==self.frame_id
                and (evidence or {}).get('source_step_id')==step['step_id']
                and pid.get('lane_change_command_id')==command_id
                and pid.get('lane_change_completed') is False
                and isinstance(pid.get('target_lane_id'),int) and not isinstance(pid.get('target_lane_id'),bool)
                and pid.get('target_lane_id')==pid.get('current_lane_id')
                and self._same_lane(previous.get('original_lane_ref'),result.get('original_lane_ref'))):
            result.update(status='DIRECTION_RESOLVED',direction=direction,
                reason='continue_issued_return_until_centered')
        return result

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
            return self._continue_return_settling(result,step,evidence)
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
        topology = pid.get('lane_predecessor_refs') or {}
        fresh_topology = ((evidence or {}).get('observation_frame_id') == self.frame_id
            and (evidence or {}).get('source_step_id') == step['step_id']
            and pid.get('lane_topology_valid') is True)
        if fresh_topology:
            matches = [name for name in ('current', 'left', 'right')
                if self._pid_lane_ref(pid, name) is not None
                and any(self._same_lane(self._lane_ref(ref), original)
                    for ref in topology.get(name, []))]
            if matches == ['current']:
                result.update(status='ALREADY_IN_ORIGINAL_LANE',
                    reason='verified_original_corridor_current')
                return self._continue_return_settling(result,step,evidence)
            if len(matches) == 1 and matches[0] in ('left', 'right'):
                direction = matches[0]
                if pid.get(direction+'_lane_change_allowed') is not True:
                    result.update(status='UNREACHABLE',reason='original_lane_change_not_legal')
                    return result
                result.update(status='DIRECTION_RESOLVED',direction=direction,
                    reason='verified_original_corridor_'+direction)
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

    def enforce_execution(self,decision,canonical,*,risk=None):
        """A later liveness or sequence controller cannot bypass plan readiness."""
        if canonical.get('decision_status')=='READY':
            result=deepcopy(decision)
            if canonical.get('deferred_lane_change') is True:
                result.update(target_lane=None,target_location=None,deferred_lane_change=True)
                if not result.get('emergency') and result.get('action') not in ('stop','emergency_brake'):
                    speed=result.get('target_speed_kmh',0.)
                    speed=float(speed) if isinstance(speed,(int,float)) and not isinstance(speed,bool) and math.isfinite(speed) else 0.
                    result.update(action='keep_lane',target_speed_kmh=max(0.,min(speed,canonical['target_speed_kmh'])),
                        longitudinal_control_mode='CAUTIOUS_SPEED_CAP',reason='lane_change_deferred_current_lane_cruise')
                for key in ('longitudinal_sequence_schema','target_acceleration_mps2','sequence_valid_until_s'):
                    result.pop(key,None)
            if (canonical.get('action') in ('turn_left','turn_right')
                    and result.get('decision_status','READY')=='READY'
                    and (result.get('action') not in ('stop','emergency_brake','decelerate')
                         or (result.get('action')=='decelerate' and risk is not None
                             and cautious_permission(risk,frame_id=canonical.get('frame_id')) is not None))
                    and not result.get('emergency',False)):
                result.update(action=canonical['action'],
                    target_location=deepcopy(canonical['target_location']))
                if result!=decision:result['reason']='active_turn_target_preserved'
            return result,result!=decision
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

    def terminate(self, request_id, reason):
        """Mirror a matching dispatcher failure without granting task completion."""
        if not self.document or self.document['request_id'] != request_id or self.state is None:
            return False
        if self.state.get('plan_status') in {'COMPLETED','FAILED','CANCELLED'}:
            return False
        self.state=deepcopy(self.state)
        index=self.state.get('active_step_index')
        if index is not None:
            self.state['step_states'][index].update(status='FAILED',reason_codes=[reason],
                terminal_frame_id=self.frame_id)
        self.state.update(plan_status='FAILED',active_step_index=None,active_step_id=None,
            reason_codes=[reason],last_frame_id=self.frame_id,
            revision=self.state.get('revision',0)+1)
        self.feedback=None
        self.risk_pause=False
        return True

    def prepare(self,document,*,frame_id,timestamp_s,speed_mps,execution_state=None,
                alignment=None,readiness=None):
        self.physical_lane_arrival=False
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
            self.speed_adjustment_seen=None
            self.last_speed_completion_evidence=None
            self.plan_origin_lane=self._pid_lane_ref((execution_state or {}).get('pid') or {},'current')
        self.frame_id=frame_id;self.feedback=None
        self.timestamp_s=timestamp_s
        dt=None if self.previous_speed_time is None else timestamp_s-self.previous_speed_time
        acceleration=(speed_mps-self.previous_observed_speed)/dt if (
            self.previous_observed_speed is not None and dt is not None and 0<dt<=.25)else None
        self.previous_observed_speed=speed_mps;self.previous_speed_time=timestamp_s
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
            self.speed_adjustment_seen=None
            self.completion_step=step['step_id'];self.completion_lane=None
            self.junction_entered=False;self.junction_heading=None;self.settled_since=None
        if not continuous:self.settled_since=None
        self.completion_time=timestamp_s;self.completion_frame=frame_id
        if kind is None:
            if action=='SET_SPEED' and any(parameters.get(key) is not None
                    for key in ('target_speed_mps','target_speed_kmh')):
                kind='TARGET_SPEED_REACHED'
            elif index+1<len(steps):
                kind={'STOP':'VEHICLE_STOPPED','EMERGENCY_BRAKE':'VEHICLE_STOPPED',
                    'ADJUST_SPEED':'TARGET_SPEED_REACHED','SET_SPEED':'TARGET_SPEED_REACHED',
                    'CHANGE_LANE':'LANE_CHANGE_COMPLETED','KEEP_LANE':'ACTION_REACHED'}.get(action)
        satisfied=False
        constrained_completion=False
        qualitative_deceleration=False
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
                    and self.speed_constraint_codes
                    and set(self.speed_constraint_codes)<= {
                        'road_speed_limit','route_speed_cap','rainy_night_speed_cap'}
                    and isinstance(effective,(int,float)) and not isinstance(effective,bool)
                    and math.isfinite(float(effective)) and float(effective)>=0.):
                constrained=float(effective)/3.6
                if target is None or constrained<float(target):
                    target=constrained
                    constrained_completion=True
            if target is not None:
                if (self.settled_target_speed_mps is None
                        or abs(self.settled_target_speed_mps-float(target))>1./3.6):
                    self.settled_since=None
                self.settled_target_speed_mps=float(target)
            satisfied=(self.feedback is None and target is not None
                and abs(speed_mps-float(target))<=self.speed_completion_tolerance_kmh/3.6)
            if (not satisfied and self.feedback is None and fresh and active_speed_step):
                current_step_state=self.state['step_states'][index]
                qualitative_deceleration=qualitative_deceleration_reached(
                    step,current_step_state,speed_mps*3.6,self.speed_completion_tolerance_kmh)
                satisfied=qualitative_deceleration
            if (self.safety_adjusted_completion and not satisfied and self.feedback is None
                    and fresh and target is not None and active_speed_step):
                adjustment=safety_adjusted_speed_evidence(evidence,frame_id=frame_id,
                    timestamp_s=timestamp_s,requested_kmh=float(target)*3.6,
                    actual_kmh=speed_mps*3.6,tolerance_kmh=self.speed_completion_tolerance_kmh,
                    acceleration_mps2=acceleration)
                if adjustment is not None:
                    satisfied=True;self.speed_adjustment_seen=adjustment
        elif kind=='LANE_CHANGE_COMPLETED':
            evidence=execution_state or {}
            pid=evidence.get('pid',{})
            from .maneuver_evidence import issued_lane_arrival
            self.physical_lane_arrival=issued_lane_arrival(
                document['request_id'],step['step_id'],evidence,frame_id)
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
            if action in ('KEEP_LANE','PROCEED') and not parameters and kind=='ACTION_REACHED':
                if self.completion_lane is None:self.completion_lane=lane
                satisfied=(centered and lane is not None and lane==self.completion_lane
                    and pid.get('in_junction') is False and speed_mps>.5)
                if lane!=self.completion_lane:self.completion_lane=lane;self.settled_since=None
            elif action=='KEEP_LANE' and set(parameters)<= {'direction'}:
                direction=str(parameters.get('direction','')).upper()
                current=self._pid_lane_ref(pid,'current')
                side_key=direction.lower()+'_lane_ref'
                positioned=(direction=='CURRENT' or (direction in ('LEFT','RIGHT')
                    and pid.get('lane_topology_valid') is True
                    and side_key in pid and pid[side_key] is None))
                if self.completion_lane is None:self.completion_lane=current
                satisfied=(centered and current is not None and positioned
                    and self._same_lane(current,self.completion_lane)
                    and pid.get('in_junction') is False and speed_mps>.5)
                if not self._same_lane(current,self.completion_lane):
                    self.completion_lane=current;self.settled_since=None
            elif action=='PROCEED' and parameters.get('condition')=='FINISH' and kind=='ACTION_REACHED':
                goal=evidence.get('navigation_goal') or {}
                values=[goal.get(key) for key in ('route_end_m','route_progress_m','endpoint_distance_m')]
                finite=all(isinstance(value,(int,float)) and not isinstance(value,bool)
                    and math.isfinite(value) for value in values)
                satisfied=(centered and finite and goal.get('valid') is True
                    and goal.get('source')=='measured_ego_public_route_endpoint'
                    and goal.get('frame_id')==frame_id and goal.get('reached') is True
                    and values[0]>0 and values[1]>=values[0]-3. and 0<=values[2]<=3.)
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
        eligible_status=(self.state or {}).get('step_states',[])
        eligible_status=eligible_status[index]['status'] if eligible_status else None
        if (self.state is None or self.state.get('plan_status')!='ACTIVE'
                or (eligible_status!='ACTIVE' and not (
                    eligible_status=='WAITING' and self.physical_lane_arrival))):satisfied=False
        if self.feedback is not None and self.feedback.get('outcome')=='FAILED':
            satisfied=False
        elif satisfied:
            if self.settled_since is None:self.settled_since=timestamp_s
        else:
            self.settled_since=None;self.speed_adjustment_seen=None
        # A single matching sample is not physical completion.  The same
        # configured hold is required for numeric targets and qualitative
        # speed bounds; a transient matching observation cannot advance a step.
        required_hold=self.speed_completion_hold_s if kind=='TARGET_SPEED_REACHED' else .5
        if (action in ('TURN','U_TURN') and kind=='JUNCTION_EXITED'
                and index+1<len(steps) and steps[index+1].get('action')=='CHANGE_LANE'):
            # Separate junction exit from the next lateral maneuver so its
            # independent origin can be observed before lateral motion starts.
            required_hold=1.25
        if action=='PROCEED' and parameters.get('condition')=='FINISH' and kind=='ACTION_REACHED':
            required_hold=0.
        if self.settled_since is not None and timestamp_s-self.settled_since>=required_hold:
            reasons=['observed_physical_completion']
            if kind=='TARGET_SPEED_REACHED' and qualitative_deceleration:
                reasons.append('observed_qualitative_deceleration_upper_bound')
            elif kind=='TARGET_SPEED_REACHED' and self.speed_adjustment_seen:
                reasons=['observed_safety_adjusted_speed_completion']+self.speed_adjustment_seen['constraint_codes']
            elif kind=='TARGET_SPEED_REACHED' and constrained_completion:
                reasons=['observed_constrained_speed_completion']+self.speed_constraint_codes
            elif action=='KEEP_LANE' and parameters.get('direction') in ('LEFT','RIGHT','CURRENT'):
                reasons.append('observed_lane_position_'+parameters['direction'].lower())
            elif self.return_resolution and self.return_resolution.get('status')=='ALREADY_IN_ORIGINAL_LANE':
                reasons=['already_in_original_lane']
            self.feedback=dict(schema_version='1.0.0',request_id=document['request_id'],
                frame_id=frame_id,step_id=step['step_id'],outcome='COMPLETED',
                reason_codes=list(dict.fromkeys(reasons)))
            if self.speed_adjustment_seen and kind=='TARGET_SPEED_REACHED':
                self.last_speed_completion_evidence=deepcopy(self.speed_adjustment_seen)
            if kind=='LANE_CHANGE_COMPLETED':
                self._record_lane_change(step,evidence,True)
            if (kind=='LANE_CHANGE_COMPLETED' and (evidence.get('pid') or {}).get('lane_change_completed') is True) or (
                    self.junction_entered and kind in ('ACTION_REACHED','JUNCTION_EXITED')
                    and action in ('TURN','U_TURN','PROCEED')):
                event='LANE_CHANGE_COMPLETED' if kind=='LANE_CHANGE_COMPLETED' else 'JUNCTION_COMPLETED'
                identity=(document['request_id'],step['step_id'],event)
                if not any((fact['request_id'],fact['step_id'],fact['condition'])==identity
                        for fact in self.completed_maneuver_facts):
                    self.completed_maneuver_facts.append(dict(request_id=document['request_id'],
                        step_id=step['step_id'],condition=event,frame_id=frame_id,
                        observed_at_s=timestamp_s,source='held_physical_execution_feedback'))
                    self.completed_maneuver_facts=self.completed_maneuver_facts[-128:]
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
        effective=self.effective_document or self.document
        active_index=0 if self.state is None else self.state.get('active_step_index')
        active_step=effective['intent']['steps'][active_index] if active_index is not None else {}
        condition_step=(active_step.get('action') in {'WAIT','CHECK','CONFIRM'}
            and ((readiness or {}).get(active_step.get('step_id')) or {}).get('satisfied') is True)
        bounded=((active_step.get('action') in FORWARD_STEPS or condition_step)
                 and cautious_permission(risk,frame_id=world['frame_id']) is not None)
        from .task_feasibility import task_permission
        bounded=bounded or task_permission(risk,active_step.get('action'),frame_id=world['frame_id'])
        if (risk.get('risk_level')!='low' and not bounded
                and not getattr(self,'physical_lane_arrival',False)):
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
                clear=(risk.get('frame_id')==world['frame_id'] and (bounded or (
                    risk.get('risk_level')=='low' and risk.get('recommended_action')=='maintain_speed'))
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
            if (step.get('action') in {'WAIT','CHECK','CONFIRM'}
                    and (step.get('parameters') or {}).get('condition')):
                # An active predicate is not permission to drive. Keep it
                # observable so fresh evidence can still complete the step.
                state['reason_codes']=['await_verified_wait_condition']
                emergency=decision.get('action')=='emergency_brake' or bool(decision.get('emergency'))
                decision.update(decision_status='BLOCKED',action='emergency_brake' if emergency else 'stop',
                    target_speed_kmh=0.,target_lane=None,target_location=None,emergency=emergency,
                    reason='await_verified_wait_condition',blocked_reason_codes=['await_verified_wait_condition'])
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
                    satisfied=task_permission(risk,'PATH_CLEAR',frame_id=world['frame_id'])
                    if not risk.get('task_feasibility'):
                        satisfied=(risk.get('frame_id')==world['frame_id'] and risk.get('risk_level')=='low')
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
        if index is not None and (effective['intent']['steps'][index].get('action')=='OVERTAKE'
                or effective['intent']['steps'][index].get('action')=='PASS_BY'
                and (effective['intent']['steps'][index].get('parameters') or {}).get('condition')=='CYCLIST'):
            current=self._pid_lane_ref((self.last_execution_state or {}).get('pid') or {},'current')
            from .maneuver_evidence import completed_lane_origin
            held_origin=completed_lane_origin(self.document,self.completed_maneuver_facts,
                self.last_execution_state,world['frame_id'])
            decision['passing_lane_established']=any(
                item.get('request_id')==self.document['request_id'] and not item.get('returned')
                and self._same_lane(item.get('target'),current) for item in self.lane_change_history) or held_origin is not None
        self.state=state
        return decision

    def completed_navigation_step(self):
        """Retire a completed maneuver without retiring the cruise mission."""
        if not self.document or (self.state or {}).get('plan_status') != 'COMPLETED':
            return None
        terminal = self.document['intent']['steps'][-1]
        if terminal['action'] not in {
                'TURN', 'CHANGE_LANE', 'KEEP_LANE', 'PROCEED', 'SET_SPEED',
                'ADJUST_SPEED', 'FOLLOW', 'OVERTAKE', 'RESUME'}:
            return None
        return dict(step_id='navigation_after_completion', action='KEEP_LANE',
                    parameters={}, trigger={'type': 'IMMEDIATE'}, depends_on=[],
                    preconditions=[], on_blocked='SAFE_STOP')

    def deferred_lane_navigation(self, decision, risk, corridor_evidence, *, maneuver_started=False,
                                 cruise_speed_kmh=None):
        """Wait for the target lane while continuing in a verified current lane."""
        if maneuver_started:return decision
        index=(self.state or {}).get('active_step_index')
        if index is None or (self.state or {}).get('plan_status')!='ACTIVE':return decision
        step=(self.effective_document or self.document)['intent']['steps'][index]
        parameters=step.get('parameters') or {}
        if (step.get('action')!='CHANGE_LANE' or parameters.get('direction') not in {'LEFT','RIGHT'}
                or self.state['step_states'][index].get('status')!='WAITING'
                or decision.get('emergency') or decision.get('action')=='emergency_brake'):
            return decision
        policy=risk.get('behavior_policy') or {}
        observed=corridor_evidence or {}
        observed_at=observed.get('observed_at_s');valid_until=observed.get('valid_until_s')
        cap=policy.get('max_speed_kmh')
        if cap is None:cap=policy.get('speed_cap_kmh')
        permission=cautious_permission(risk,frame_id=self.frame_id)
        if permission is not None:cap=permission.get('max_speed_kmh',cap)
        ready=(risk.get('frame_id')==self.frame_id and not policy.get('physical_stop')
            and policy.get('allow_positive_acceleration') is not False
            and policy.get('disposition') in {'CLEAR','CAUTIOUS_PROCEED'}
            and observed.get('request_id')==self.document['request_id']
            and observed.get('frame_id')==self.frame_id and observed.get('condition') in {'PATH_CLEAR','CURRENT_LANE_MOTION'}
            and observed.get('valid') is True and observed.get('satisfied') is True
            and observed.get('source')=='fresh_lidar_radar_driving_corridor'
            and isinstance(observed_at,(int,float)) and not isinstance(observed_at,bool)
            and math.isfinite(observed_at) and 0<=self.timestamp_s-observed_at<=.25
            and isinstance(valid_until,(int,float)) and not isinstance(valid_until,bool)
            and math.isfinite(valid_until) and self.timestamp_s<=valid_until
            and isinstance(cap,(int,float)) and not isinstance(cap,bool) and math.isfinite(cap) and cap>=5.)
        if not ready:return decision
        target_speed = min(10., cap)
        nominal = cruise_speed_kmh
        steps = self.document['intent']['steps']
        explicit_bound = None
        for prior_index, prior in enumerate(steps[:index]):
            if self.state['step_states'][prior_index].get('status') != 'COMPLETED':
                continue
            prior_parameters = prior.get('parameters') or {}
            if prior.get('action') not in {'SET_SPEED','ADJUST_SPEED'}:
                continue
            explicit = prior_parameters.get('target_speed_kmh')
            if prior_parameters.get('target_speed_mps') is not None:
                explicit = prior_parameters['target_speed_mps'] * 3.6
            if (isinstance(explicit,(int,float)) and not isinstance(explicit,bool)
                    and math.isfinite(explicit) and explicit >= 0.):
                explicit_bound = explicit
        if explicit_bound is not None:
            target_speed = min(target_speed, explicit_bound)
            if isinstance(nominal,(int,float)) and not isinstance(nominal,bool):
                nominal = min(nominal, explicit_bound)
        # Waiting for a legal merge need not prevent closing the gap to a
        # moving referent. A faster approach needs braking-distance evidence.
        if (isinstance(nominal, (int,float)) and not isinstance(nominal,bool)
                and math.isfinite(nominal) and nominal > 10.):
            from lightweight_vla_adapter.src.risk_behavior import stopping_clearance
            candidate = min(30., cap, nominal)
            lidar = policy.get('lidar') or {}
            radar = risk.get('physical_forward_radar') or {}
            clearance = lidar.get('clear_distance_m')
            required = stopping_clearance(candidate)
            nearest = radar.get('nearest_distance_m')
            if (not policy.get('absolute_stop_active') and lidar.get('valid') is True
                    and lidar.get('sensor_frame') == radar.get('sensor_frame') == observed.get('sensor_frame')
                    and lidar.get('sensor_frame') is not None
                    and lidar.get('observed_at_s') == observed_at
                    and isinstance(clearance,(int,float)) and not isinstance(clearance,bool)
                    and math.isfinite(clearance) and clearance >= required
                    and (radar.get('obstacle_candidate_count') == 0 and nearest is None
                         or isinstance(nearest,(int,float)) and not isinstance(nearest,bool)
                         and math.isfinite(nearest) and nearest > required)):
                target_speed = candidate
        result=deepcopy(decision)
        result.update(decision_status='READY',action='keep_lane',target_lane=None,target_location=None,
            target_speed_kmh=target_speed,emergency=False,allow_positive_acceleration=True,
            longitudinal_control_mode='CAUTIOUS_SPEED_CAP',reason='lane_change_deferred_current_lane_cruise',
            deferred_lane_change=True,blocked_reason_codes=decision.get('blocked_reason_codes',[]),
            deferred_current_lane_evidence=deepcopy(observed))
        if permission is not None:result['risk_motion_permission']=deepcopy(permission)
        for key in ('longitudinal_sequence_schema','target_acceleration_mps2','sequence_valid_until_s'):
            result.pop(key,None)
        return result

    def start_authorized_lane_motion(self, decision, risk, *, cruise_speed_kmh=None):
        """A safe lane change needs a rolling target after a preceding stop."""
        action=decision.get('action','')
        direction=action.removeprefix('lane_change_')
        if (direction not in {'left','right'} or decision.get('source_step_action')!='CHANGE_LANE'
                or decision.get('decision_status')!='READY' or decision.get('emergency')
                or decision.get('blocked_reason_codes') or risk.get('frame_id')!=self.frame_id):return decision
        from .task_feasibility import task_permission
        contract=risk.get('target_lane_contract') or {}
        policy=risk.get('behavior_policy') or {}
        cap=(risk.get('lane_change',{}).get(direction) or {}).get('speed_cap_kmh')
        motion_cap=policy.get('max_speed_kmh')
        target=decision.get('target_speed_kmh')
        rolling_target=5.
        if (isinstance(cruise_speed_kmh,(int,float)) and not isinstance(cruise_speed_kmh,bool)
                and math.isfinite(cruise_speed_kmh) and cruise_speed_kmh>0.
                and isinstance(cap,(int,float)) and not isinstance(cap,bool)
                and math.isfinite(cap) and cap>=15.):
            rolling_target=min(15.,cruise_speed_kmh)
            index=(self.state or {}).get('active_step_index')
            if self.document is not None and isinstance(index,int):
                for prior_index,step in enumerate(self.document['intent']['steps'][:index+1]):
                    if (prior_index<index and self.state['step_states'][prior_index].get('status')!='COMPLETED'):
                        continue
                    params=step.get('parameters') or {}
                    explicit=params.get('target_speed_kmh')
                    if params.get('target_speed_mps') is not None:
                        explicit=params['target_speed_mps']*3.6
                    if (isinstance(explicit,(int,float)) and not isinstance(explicit,bool)
                            and math.isfinite(explicit) and explicit>=0.):
                        rolling_target=min(rolling_target,explicit)
        allowed=(contract.get('direction')==direction and contract.get('can_execute') is True
            and policy.get('allow_positive_acceleration') is not False
            and policy.get('disposition') in {'CLEAR','CAUTIOUS_PROCEED'} and not policy.get('physical_stop')
            and not policy.get('absolute_stop_active')
            and task_permission(risk,'CHANGE_LANE_'+direction.upper(),frame_id=self.frame_id)
            and task_permission(risk,'PATH_CLEAR',frame_id=self.frame_id)
            and isinstance(cap,(int,float)) and not isinstance(cap,bool) and math.isfinite(cap) and cap>=5.
            and isinstance(motion_cap,(int,float)) and not isinstance(motion_cap,bool)
                and math.isfinite(motion_cap) and motion_cap>=1.
            and isinstance(target,(int,float)) and not isinstance(target,bool) and math.isfinite(target) and 0<=target<rolling_target)
        if not allowed:return decision
        result=deepcopy(decision)
        result.update(target_speed_kmh=min(rolling_target,cap,motion_cap),
            reason='sensor_authorized_low_speed_lane_departure',allow_positive_acceleration=True,
            longitudinal_control_mode='CAUTIOUS_SPEED_CAP')
        for key in ('longitudinal_sequence_schema','target_acceleration_mps2','sequence_valid_until_s'):
            result.pop(key,None)
        return result

    def navigation_handoff(self, decision, risk, *, cruise_speed_kmh):
        step = self.completed_navigation_step()
        if (step is None or not str(decision.get('reason', '')).startswith('plan_completed')
                or decision.get('emergency') or decision.get('decision_status') != 'READY'):
            return decision
        if (risk.get('frame_id') != self.frame_id or risk.get('risk_level') != 'low'
                and cautious_permission(risk, frame_id=self.frame_id) is None):
            return decision
        if (isinstance(cruise_speed_kmh, bool) or not isinstance(cruise_speed_kmh, (int, float))
                or not math.isfinite(cruise_speed_kmh) or not 0 < cruise_speed_kmh <= 100):
            return decision
        result = dict(decision, action='keep_lane', target_speed_kmh=float(cruise_speed_kmh),
                      target_lane=None, target_location=None, emergency=False,
                      source_step_id=step['step_id'], source_step_action='KEEP_LANE',
                      reason='completed_plan_navigation_handoff')
        for key in ('target_acceleration_mps2', 'longitudinal_sequence_schema',
                    'sequence_valid_until_s'):
            result.pop(key, None)
        return result

    def _observe_condition_completion(self,world,risk,readiness=None):
        """Only explicit, freshly observed predicates complete a check/wait."""
        index=None if self.state is None else self.state.get('active_step_index')
        effective=self.effective_document or self.document
        step=effective['intent']['steps'][index] if index is not None else None
        condition=(step.get('parameters') or {}).get('condition') if step else None
        unbound_distance=(step is not None and step['action']=='KEEP_SAFE_DISTANCE'
            and not any(step.get(key) for key in ('target','target_ref'))
            and not any((step.get('parameters') or {}).get(key) for key in ('target','target_ref','condition')))
        if unbound_distance:condition='PATH_CLEAR'
        eligible=(step is not None and self.state.get('plan_status')=='ACTIVE'
                  and (step['action'] in ('CHECK','WAIT','CONFIRM','YIELD')
                       or step['action'] in ('OVERTAKE','PASS_BY') and condition is not None
                       or unbound_distance))
        fresh=(risk.get('frame_id')==world['frame_id']==self.frame_id)
        safe=(risk.get('risk_level')=='low' or cautious_permission(risk,frame_id=world['frame_id']) is not None)
        from .task_feasibility import task_permission
        safe=safe or task_permission(risk,'PATH_CLEAR',frame_id=world['frame_id'])
        clear=fresh and safe and not self.risk_pause
        if condition in ('LEFT_LANE_SAFE','RIGHT_LANE_SAFE'):
            direction=condition.split('_')[0].lower()
            clear=clear and risk.get('lane_change',{}).get(direction,{}).get('is_safe') is True
        elif condition!='PATH_CLEAR':
            clear=False
            if eligible:
                observed=(readiness or {}).get(step['step_id']) or {}
                if step.get('action')=='CONFIRM' and condition in {
                        'JUNCTION_COMPLETED','LANE_CHANGE_COMPLETED'}:
                    binding=(self.document['request_id'],step['step_id'])
                    if binding not in self.maneuver_confirmation_bindings:
                        parameters=step.get('parameters') or {}
                        matches=[fact for fact in self.completed_maneuver_facts
                            if fact['condition']==condition and fact['observed_at_s']<=self.timestamp_s
                            and (not parameters.get('source_request_id')
                                or fact['request_id']==parameters['source_request_id'])
                            and (not parameters.get('source_step_id')
                                or fact['step_id']==parameters['source_step_id'])]
                        self.maneuver_confirmation_bindings[binding]=deepcopy(matches[-1]) if matches else None
                        while len(self.maneuver_confirmation_bindings)>128:
                            self.maneuver_confirmation_bindings.pop(next(iter(self.maneuver_confirmation_bindings)))
                    fact=self.maneuver_confirmation_bindings[binding]
                    observed=dict(request_id=self.document['request_id'],step_id=step['step_id'],
                        condition=condition,frame_id=world['frame_id'],valid=bool(fact),
                        satisfied=bool(fact),source='recorded_physical_maneuver_fact',
                        observed_at_s=self.timestamp_s,valid_until_s=self.timestamp_s+.25,
                        completed_event_fact=deepcopy(fact))
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
                clear=(fresh and safe and not self.risk_pause
                    and self._fresh_step_evidence(observed,step,world['frame_id'])
                    and observed.get('condition')==condition
                    and explicit and observed.get('valid',True) is True
                    and isinstance(observed.get('source'),str) and bool(observed.get('source'))
                    and time_valid and expiry_valid)
        key=(self.document['request_id'],step['step_id'],condition) if eligible else None
        continuous=(self.condition_time is not None and self.timestamp_s is not None
                    and 0 < self.timestamp_s-self.condition_time<=.25
                    and self.condition_frame!=self.frame_id)
        if (key!=self.condition_key or not continuous or not eligible or not clear
                or self.condition_since is None):
            self.condition_since=self.timestamp_s if eligible and clear else None
        self.condition_key=key;self.condition_time=self.timestamp_s;self.condition_frame=self.frame_id
        if self.condition_since is not None and self.timestamp_s-self.condition_since>=.5:
            self.feedback=dict(schema_version='1.0.0',request_id=self.document['request_id'],
                frame_id=self.frame_id,step_id=step['step_id'],outcome='COMPLETED',
                reason_codes=['observed_condition_completed'])
            self.condition_since=None
