"""Recover passing context only from a held, issued lane-change completion."""
import math


def issued_lane_arrival(request_id, step_id, execution, frame_id):
    """An issued maneuver's measured arrival is a fact, not a new permission."""
    execution = execution or {}
    if (execution.get('observation_frame_id') != frame_id
            or execution.get('source_step_id') != step_id):
        return False
    pid = execution.get('pid') or {}
    current = pid.get('current_lane_ref') or {}
    numeric = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    return (pid.get('lane_change_completed') is True
            and not pid.get('emergency_latched')
            and pid.get('lane_change_command_id') == request_id + ':' + step_id
            and all(isinstance(current.get(key), int) and not isinstance(current.get(key), bool)
                    for key in ('road_id', 'section_id', 'lane_id'))
            and current['lane_id'] == pid.get('target_lane_id')
            and numeric(pid.get('lateral_error_m')) and abs(pid['lateral_error_m']) <= .35
            and numeric(pid.get('heading_error_deg')) and abs(pid['heading_error_deg']) <= 5.)


def completed_lane_origin(document, facts, execution, frame_id):
    if not document or (execution or {}).get('observation_frame_id') != frame_id:
        return None
    pid=(execution or {}).get('pid') or {}
    current=pid.get('current_lane_ref') or {}
    numeric=lambda value:isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
    if (pid.get('lane_change_completed') is not True or pid.get('emergency_latched')
            or not all(isinstance(current.get(key),int) and not isinstance(current.get(key),bool)
                       for key in ('road_id','section_id','lane_id'))
            or current['lane_id'] != pid.get('target_lane_id')
            or not numeric(pid.get('lateral_error_m')) or abs(pid['lateral_error_m']) > .35
            or not numeric(pid.get('heading_error_deg')) or abs(pid['heading_error_deg']) > 5.):
        return None
    request=document.get('request_id')
    steps={item['step_id']:item for item in document['intent']['steps']}
    for fact in reversed(facts):
        step=steps.get(fact.get('step_id'))
        if (fact.get('request_id') != request
                or fact.get('condition') != 'LANE_CHANGE_COMPLETED'
                or fact.get('source') != 'held_physical_execution_feedback'
                or not step or step.get('action') != 'CHANGE_LANE'
                or pid.get('lane_change_command_id') != request+':'+step['step_id']):
            continue
        parameters=step.get('parameters') or {}
        direction=parameters.get('direction')
        if parameters.get('return_to') is not None or direction not in {'LEFT','RIGHT'}:
            continue
        return dict(direction=direction.lower(),origin_side='RIGHT' if direction=='LEFT' else 'LEFT',
            source='held_physical_completion_and_current_latched_lane',
            source_step_id=step['step_id'],frame_id=frame_id)
    return None
