"""Keep risk classifications separate from evidence-bounded motion authority."""
import math
from collections import deque
import numpy as np
from .route_projection_compiled import project_points_to_route, projection_available


FORWARD_STEPS = {'KEEP_LANE', 'SET_SPEED', 'ADJUST_SPEED', 'FOLLOW', 'TURN', 'PROCEED',
                 'OVERTAKE', 'PASS_BY', 'KEEP_SAFE_DISTANCE'}
FORWARD_INTENTS = FORWARD_STEPS | {'TURN_LEFT', 'TURN_RIGHT', 'ACCELERATE', 'DECELERATE', 'RESUME',
    'CHANGE_LANE_LEFT', 'CHANGE_LANE_RIGHT', 'YIELD'}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def resume_bounded_navigation(decision, risk, cruise_speed_kmh):
    permission = cautious_permission(risk, frame_id=decision.get('frame_id'))
    source_action = decision.get('source_step_action')
    established_pass=(source_action in {'OVERTAKE','PASS_BY'} and decision.get('passing_lane_established') is True)
    if (not permission or permission['schema_version'] != 'cautious_motion/2.0'
            or decision.get('decision_status') != 'READY'
            or decision.get('action') not in ({'keep_lane','accelerate'} if established_pass
                                             else {'keep_lane', 'turn_left', 'turn_right'})
            or source_action not in {'KEEP_LANE', 'TURN', 'PROCEED', 'PASS_BY', 'KEEP_SAFE_DISTANCE'}
                and not established_pass
            or decision.get('blocked_reason_codes') or decision.get('emergency')):
        return dict(decision)
    if (not isinstance(cruise_speed_kmh, (int, float)) or isinstance(cruise_speed_kmh, bool)
            or not math.isfinite(cruise_speed_kmh) or cruise_speed_kmh < 0):
        return dict(decision)
    result = dict(decision)
    result['target_speed_kmh'] = min(cruise_speed_kmh, permission['max_speed_kmh'])
    result['reason'] = 'sensor_bounded_navigation_cruise'
    return result


def stopping_clearance(speed_kmh):
    speed = max(0., speed_kmh) / 3.6
    return 2.5 + speed * .65 + speed * speed / 6.


def clearance_speed_cap(distance_m, maximum_kmh):
    if not finite(distance_m) or distance_m <= 3.:
        return 0.
    speed = (-.65 + math.sqrt(.65**2 + 4. / 6. * (distance_m - 3.))) * 3.
    return min(maximum_kmh, math.floor(speed * 3.6 * 10.) / 10.)


def visual_only_high(risk):
    reasons = set(risk.get('reason_codes') or [])
    provenance = risk.get('prediction_provenance') or {}
    return (risk.get('risk_level') == 'high'
        and risk.get('source') in {'learned_raw_camera_visual_risk_head',
                                  'learned_raw_camera_visual_risk_head+risk_continuity'}
        and provenance.get('risk_residual_applied',provenance.get('residual_applied')) is False
        and bool(reasons & {'learned_visual_risk_high', 'temporal_visual_risk_high'})
        and reasons <= {'learned_visual_risk_high', 'temporal_visual_risk_high',
                        'physical_forward_radar_caution_distance'})


def lidar_corridor_evidence(points, route, *, half_width_m, ego_front_m, sensor_frame, observed_at_s):
    """Require measured road support, not just absence of obstacle returns.

    Heights use planner road geometry in world coordinates, so slopes do not
    share a fixed LiDAR-height threshold. Unobserved bins end the free horizon.
    """
    info = dict(schema_version='lidar_route_clearance/1.0', valid=False,
                source='same_frame_lidar_and_static_road_geometry', sensor_frame=sensor_frame,
                observed_at_s=observed_at_s, clear_distance_m=0., nearest_obstacle_m=None)
    if points is None or route is None or not finite(half_width_m) or not finite(ego_front_m):
        return dict(info, reason='lidar_or_route_unavailable')
    xyz = np.asarray(points, dtype=float)
    road = np.asarray(route, dtype=float)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 20 or not np.isfinite(xyz).all()
            or road.ndim != 2 or road.shape[1] != 3 or len(road) < 2
            or not np.isfinite(road).all() or half_width_m <= 0 or ego_front_m <= 0):
        return dict(info, reason='invalid_lidar_or_geometry')
    lengths = np.linalg.norm(np.diff(road[:, :2], axis=0), axis=1)
    if (lengths < .01).any():
        return dict(info, reason='degenerate_route_geometry')
    if (np.abs(np.diff(road[:, 2])) / lengths > .3).any():
        return dict(info, reason='implausible_route_surface_grade')
    xyz = xyz[np.linalg.norm(xyz[:, :2] - road[0, :2], axis=1) <= 65.]
    # Every relevant return is within this conservative polyline rectangle.
    radius=half_width_m+.5+1e-8
    lower=road[:,:2].min(axis=0)-radius;upper=road[:,:2].max(axis=0)+radius
    xyz=xyz[((xyz[:,:2]>=lower)&(xyz[:,:2]<=upper)).all(axis=1)]
    if projection_available():
        lateral, progress, height = project_points_to_route(xyz, road, lengths)
    else:
        lateral = np.full(len(xyz), np.inf)
        progress = np.zeros(len(xyz)); height = np.zeros(len(xyz)); offset = 0.
        x, y = xyz[:, 0], xyz[:, 1]
        for first, last, length in zip(road[:-1], road[1:], lengths):
            delta = last[:2] - first[:2]
            dx, dy = x-first[0], y-first[1]
            fraction = np.clip((dx*delta[0]+dy*delta[1]) / length**2, 0., 1.)
            rx, ry = dx-fraction*delta[0], dy-fraction*delta[1]
            distance = np.sqrt(rx*rx+ry*ry)
            update = distance < lateral
            lateral[update] = distance[update]
            progress[update] = offset + fraction[update]*length
            height[update] = xyz[update, 2] - (first[2] + fraction[update]*(last[2]-first[2]))
            offset += length
    forward = progress > ego_front_m + .4
    # Inflated swept path also screens nearby entrants; no actor truth is used.
    obstacle = forward & (lateral <= half_width_m + .5) & (height > .3) & (height < 3.)
    nearest = float(progress[obstacle].min() - ego_front_m) if obstacle.any() else None
    ground = forward & (lateral <= half_width_m) & (np.abs(height) <= .25)
    available = min(60., float(lengths.sum()) - ego_front_m)
    # The roof LiDAR cannot measure road immediately beside the bonnet.
    # This near zone is covered separately by the fresh forward radar contract.
    clear = 0.; counts = []
    ground_progress = np.sort(progress[ground]-ego_front_m)
    for start in np.arange(3., max(3., available-1.), 1.):
        lower, upper = np.searchsorted(ground_progress, [start, start+1.5], side='left')
        count = int(upper-lower); counts.append(count)
        if count < 2:
            break
        clear = float(start+1.)
    if nearest is not None:
        clear = min(clear, max(0., nearest-1.))
    return dict(info, valid=True, clear_distance_m=round(clear, 3),
                nearest_obstacle_m=round(nearest, 3) if nearest is not None else None,
                road_support_counts=counts, obstacle_points=int(obstacle.sum()),
                near_field_blind_zone_m=3., near_field_requires_forward_radar=True,
                reason='measured_road_support' if clear > 0 else 'unobserved_forward_road')


def _grounded_passing_valid(risk):
    policy=risk.get('behavior_policy') or {}; proof=policy.get('grounded_passing') or {}
    lidar=policy.get('lidar') or {}; now=(risk.get('risk_evidence') or {}).get('decision_timestamp_s')
    lateral=proof.get('lateral_m'); transverse=proof.get('transverse_speed_mps')
    cyclist=proof.get('target_category')=='cyclist'
    cap=proof.get('authorized_speed_cap_kmh',30.)
    return (visual_only_high(risk) and proof.get('schema_version')=='measured_adjacent_passing/1.0'
        and proof.get('frame_id')==risk.get('frame_id') and proof.get('sensor_frame')==lidar.get('sensor_frame')
        and proof.get('policy_truth_access') is False and proof.get('target_id') is not None
        and proof.get('corridor_clear') is True and proof.get('origin_side') in {'LEFT','RIGHT'}
        and all(finite(v) for v in (now,proof.get('observed_at_s'),lateral,transverse,policy.get('high_clear_since_s')))
        and 0<=now-proof['observed_at_s']<=.25 and now-policy['high_clear_since_s']>=2.
        and 2.7<=abs(lateral)<=5. and lateral*transverse>=-.5
        and (not cyclist or proof.get('coordinate_frame')=='MEASURED_ROAD_LOCAL'
             and proof.get('velocity_valid') is True)
        and finite(cap) and (15. if cyclist else 30.)<=cap<=30.
        and not policy.get('bounded_initial_slowdown')
        and (cyclist or not policy.get('bounded_short_horizon'))
        and not policy.get('emergency_evidence') and not policy.get('physical_stop')
        and policy.get('closing_ttc_response')=='NONE'
        and finite(policy.get('raw_physical_speed_cap_kmh')) and policy['raw_physical_speed_cap_kmh']>=cap
        and finite(policy.get('recent_physical_speed_cap_kmh')) and policy['recent_physical_speed_cap_kmh']>=cap
        and lidar.get('nearest_obstacle_m') is None)


def ground_established_passing(risk, decision, proof):
    """Bound an established maneuver using fresh physical evidence, not a global class."""
    if (decision.get('source_step_action') not in {'OVERTAKE','PASS_BY'} or decision.get('passing_lane_established') is not True
            or decision.get('decision_status')!='READY' or decision.get('blocked_reason_codes')
            or decision.get('emergency') or not cautious_permission(risk,frame_id=decision.get('frame_id'))):
        return risk
    radar=risk.get('physical_forward_radar') or {}
    cyclist=(proof or {}).get('target_category')=='cyclist'
    if decision.get('source_step_action')=='PASS_BY' and not cyclist:
        return risk
    result=dict(risk);policy=dict(risk.get('behavior_policy') or {})
    cap=30.
    if cyclist:
        values=[policy.get('raw_physical_speed_cap_kmh'),policy.get('recent_physical_speed_cap_kmh')]
        if not all(finite(value) for value in values):return risk
        cap=min(30.,*values)
        distance=radar.get('nearest_distance_m')
        if finite(distance):cap=min(cap,clearance_speed_cap(distance,30.))
        elif radar.get('obstacle_candidate_count')!=0:return risk
    elif radar.get('obstacle_candidate_count')!=0 or radar.get('nearest_distance_m') is not None:
        return risk
    policy['grounded_passing']=dict(proof or {},authorized_speed_cap_kmh=cap);result['behavior_policy']=policy
    if not _grounded_passing_valid(result):return risk
    policy.update(max_speed_kmh=cap,required_clearance_m=stopping_clearance(cap),
        response_tier='TASK_GROUNDED_PASSING',nominal_speed_cap_kmh=cap)
    if cautious_permission(result,frame_id=decision.get('frame_id')) is None:return risk
    return result


def cautious_permission(risk, *, frame_id=None):
    policy = risk.get('behavior_policy') or {}
    evidence = risk.get('risk_evidence') or {}
    lidar = policy.get('lidar') or {}
    radar = risk.get('physical_forward_radar') or {}
    now = evidence.get('decision_timestamp_s')
    observed = lidar.get('observed_at_s')
    cap = policy.get('max_speed_kmh')
    version2 = policy.get('schema_version') == 'risk_behavior/2.0'
    level = risk.get('risk_level')
    allowed_high = version2 and (visual_only_high(risk) or risk.get('learned_risk_is_advisory') is True) and policy.get('visual_advisory') is True
    normal_clear = version2 and level == 'low' and policy.get('disposition') == 'CLEAR'
    limit = 60. if normal_clear else 30. if allowed_high and _grounded_passing_valid(risk) else 10. if allowed_high else 30. if version2 else 6.
    reasons = risk.get('reason_codes') or []
    distance=radar.get('nearest_distance_m')
    required=policy.get('required_clearance_m')
    radar_valid=(str(radar.get('schema_version','')).startswith('physical_front_radar/')
        and radar.get('sensor_frame') == lidar.get('sensor_frame')
        and (radar.get('obstacle_candidate_count') == 0 and distance is None
             or finite(distance) and finite(required) and distance > required))
    slowdown_only=policy.get('bounded_initial_slowdown') is True
    bounded_short=policy.get('bounded_short_horizon') is True
    if not ((level == 'medium' or allowed_high or level=='low' and bounded_short or normal_clear)
            and (normal_clear or risk.get('recommended_action') == 'monitor')
            and policy.get('schema_version') in ('risk_behavior/1.0', 'risk_behavior/2.0')
            and policy.get('disposition') in ('CAUTIOUS_PROCEED', 'CLEAR') and policy.get('confirmed') is True
            and policy.get('frame_id') == risk.get('frame_id')
            and (frame_id is None or frame_id == risk.get('frame_id'))
            and evidence.get('is_fresh') is True and lidar.get('valid') is True
            and radar_valid
            and lidar.get('sensor_frame') == evidence.get('sensor_frame')
            and all(finite(v) for v in (now, observed, cap, policy.get('valid_until_s')))
            and 0 <= now-observed <= .25 and now <= policy['valid_until_s']
            and 0 < cap <= limit and policy.get('required_clearance_m', math.inf) <= lidar.get('clear_distance_m', 0.)
            and stopping_clearance(cap) <= lidar.get('clear_distance_m', 0.)
            and (not allowed_high or slowdown_only or finite(policy.get('high_clear_since_s'))
                 and now - policy['high_clear_since_s'] >= 1.)
            and (not slowdown_only or (1. if bounded_short else 5.) <= cap <= 10.
                 and policy.get('allow_positive_acceleration') is False
                 and policy.get('confirmation_basis') == 'fresh_measured_corridor')
            and not any(str(c).startswith('physical_forward_radar_') and c != 'physical_forward_radar_caution_distance' for c in reasons)
            and not (finite(radar.get('nearest_closing_velocity_mps')) and radar['nearest_closing_velocity_mps'] > 0
                and finite(radar.get('nearest_closing_distance_m'))
                and radar['nearest_closing_distance_m'] / radar['nearest_closing_velocity_mps'] <= (
                    1.5 if policy.get('closing_ttc_response') == 'BOUNDED_DECELERATION' and cap <= 10. else 5.))
            and not risk.get('behavior_policy', {}).get('physical_stop')):
        return None
    permission=dict(schema_version='cautious_motion/2.0' if version2 else 'cautious_motion/1.0', source='sensor_risk_behavior_policy',
                frame_id=policy['frame_id'], sensor_frame=lidar['sensor_frame'],
                observed_at_s=observed, valid_until_s=policy['valid_until_s'], max_speed_kmh=cap)
    if slowdown_only or policy.get('closing_ttc_response') == 'BOUNDED_DECELERATION':
        permission['allow_positive_acceleration']=False
    return permission


class RiskBehaviorPolicy:
    def __init__(self, confirmation_s=.5, max_speed_kmh=30., high_speed_kmh=10., high_confirmation_s=1., clear_speed_kmh=60.):
        if not finite(confirmation_s) or not 0 < confirmation_s <= 2.:
            raise ValueError('Invalid cautious confirmation duration')
        if not finite(max_speed_kmh) or not 0 < max_speed_kmh <= 30.:
            raise ValueError('Medium risk speed must be at most 30 km/h')
        if not finite(high_speed_kmh) or not 0 < high_speed_kmh <= 10.:
            raise ValueError('Visual-only high risk speed must be at most 10 km/h')
        if not finite(high_confirmation_s) or not 1. <= high_confirmation_s <= 3.:
            raise ValueError('High risk clearance requires at least one second')
        if not finite(clear_speed_kmh) or not 0 < clear_speed_kmh <= 60.:
            raise ValueError('Clear-road speed ceiling must be at most 60 km/h')
        self.confirmation_s=confirmation_s; self.max_speed_kmh=max_speed_kmh
        self.high_speed_kmh=high_speed_kmh; self.high_confirmation_s=high_confirmation_s
        self.clear_speed_kmh=clear_speed_kmh
        self.previous_time=None; self.previous_frame=None; self.since=None
        self.previous_observed_at=None
        self.high_since=None
        self.short_horizon_active=False
        self.cap_history=deque()

    def apply(self, risk, lidar, *, frame_id, timestamp_s, ego_speed_kmh, parsed_intent, crossing_hazard=False,task_conditioned=False):
        result=dict(risk); level=risk.get('risk_level')
        radar=risk.get('physical_forward_radar') or {}; evidence=risk.get('risk_evidence') or {}
        learned_source=str(risk.get('source','')).startswith('learned_') or risk.get('source')=='event_memory_vla'
        task_advisory=bool(task_conditioned and learned_source and level=='high')
        advisory=visual_only_high(risk) or task_advisory
        if task_advisory:result['learned_risk_is_advisory']=True
        limit=self.high_speed_kmh if advisory else self.clear_speed_kmh if level == 'low' else self.max_speed_kmh
        radar_distance=radar.get('nearest_distance_m')
        observed_clearance=lidar.get('clear_distance_m',0.)
        if finite(radar_distance):observed_clearance=min(observed_clearance,radar_distance-1.)
        cap=clearance_speed_cap(observed_clearance,limit)
        raw_measured_cap=cap
        physical_cap=clearance_speed_cap(observed_clearance,self.clear_speed_kmh)
        observed=lidar.get('observed_at_s')
        history_fresh=(finite(timestamp_s) and finite(observed)
            and 0 <= timestamp_s-observed <= .25 and evidence.get('is_fresh') is True
            and lidar.get('valid') is True and lidar.get('sensor_frame') == evidence.get('sensor_frame')
            and radar.get('sensor_frame') == lidar.get('sensor_frame'))
        if history_fresh:
            if self.cap_history and observed < self.cap_history[-1][0]:
                self.cap_history.clear()
            while self.cap_history and self.cap_history[0][0] < observed-.5:
                self.cap_history.popleft()
            if not self.cap_history or observed > self.cap_history[-1][0]:
                self.cap_history.append((observed,physical_cap))
            # Past low bounds can delay recovery, never override today's lower bound.
            cap=min(cap,min(entry[1] for entry in self.cap_history))
        else:
            self.cap_history.clear()
        measured_cap=cap
        distance=lidar.get('nearest_obstacle_m'); closing=radar.get('nearest_closing_velocity_mps')
        closing_distance=radar.get('nearest_closing_distance_m')
        ttc=closing_distance/closing if finite(closing) and closing > 0 and finite(closing_distance) else None
        closing_count=radar.get('obstacle_candidate_count')
        closing_evidence_consistent=(finite(radar_distance) and isinstance(closing_count,int)
            and not isinstance(closing_count,bool) and closing_count>0)
        ttc_warning = finite(ttc) and 1.5 < ttc <= 5. and closing_evidence_consistent
        if ttc_warning:
            cap=min(cap,10.)
        required=stopping_clearance(max(ego_speed_kmh,cap,6.))
        current_required=required
        physical_stop=bool(crossing_hazard) or (finite(distance) and distance <= required) or (
            finite(ttc) and (ttc <= 1.5 or ttc <= 5. and not closing_evidence_consistent))
        emergency=bool(crossing_hazard) or (finite(ttc) and ttc <= 1.5) or (
            finite(distance) and distance <= 3.) or (finite(radar_distance) and radar_distance <= 3.)
        fresh=(finite(timestamp_s) and finite(lidar.get('observed_at_s'))
               and 0 <= timestamp_s-lidar['observed_at_s'] <= .25
               and evidence.get('is_fresh') is True and lidar.get('sensor_frame') == evidence.get('sensor_frame'))
        reasons=risk.get('reason_codes') or []
        hard=(level == 'high' or risk.get('recommended_action') == 'emergency_brake') and not advisory
        physical_caution=any(str(c).startswith('physical_forward_radar_') and c != 'physical_forward_radar_caution_distance' for c in reasons)
        if finite(radar_distance) and radar_distance <= required:
            physical_stop=True
        emergency=emergency or 'physical_forward_radar_emergency_distance' in reasons
        radar_ready=(str(radar.get('schema_version','')).startswith('physical_front_radar/')
                     and radar.get('sensor_frame') == lidar.get('sensor_frame')
                     and isinstance(radar.get('obstacle_candidate_count'),int)
                     and not isinstance(radar.get('obstacle_candidate_count'),bool)
                     and (radar.get('obstacle_candidate_count') == 0 and radar_distance is None
                          or finite(radar_distance) and radar_distance > required))
        eligible=parsed_intent in FORWARD_INTENTS
        continuous=(self.previous_time is not None and finite(timestamp_s)
                    and 0 < timestamp_s-self.previous_time <= .25)
        same_frame=lidar.get('sensor_frame') == self.previous_frame
        observed_at=lidar.get('observed_at_s')
        repeat_valid=(continuous and same_frame and finite(observed_at)
                      and finite(self.previous_observed_at)
                      and abs(observed_at-self.previous_observed_at) <= 1e-6)
        advancing=(continuous and not same_frame and finite(observed_at)
                   and finite(self.previous_observed_at) and observed_at > self.previous_observed_at)
        if same_frame and self.previous_time is not None and not repeat_valid:
            fresh=False
        cadence_ok=self.previous_time is None or advancing or repeat_valid
        # A measured short horizon is a speed bound, not an observed obstacle.
        # Brake toward a crawl while preserving the current-speed collision checks.
        short_horizon=(eligible and fresh and cadence_ok and radar_ready and lidar.get('valid') is True
            and not physical_stop and not physical_caution and not hard
            and (level in {'low','medium'} or advisory)
            and (observed_clearance < current_required or cap < 5. or self.short_horizon_active
                 and observed_clearance < stopping_clearance(self.max_speed_kmh)+1.) and cap >= 1.)
        self.short_horizon_active=short_horizon
        if short_horizon:
            cap=min(cap,10.)
            required=stopping_clearance(cap)
        clear=(fresh and radar_ready and cap >= (1. if short_horizon else min(5.,limit)) and lidar.get('valid') is True and lidar.get('clear_distance_m', 0.) >= required
               and not physical_stop and not physical_caution)
        if not clear or not eligible or hard:
            self.since=None; self.high_since=None
        elif not cadence_ok or self.since is None:
            self.since=timestamp_s
        # Clearance history belongs to sensor evidence, not classifier labels.
        # MEDIUM/HIGH alternation must not repeatedly erase verified free space.
        self.high_since=self.since if advisory else None
        confirmed=clear and eligible and self.since is not None and timestamp_s-self.since >= self.confirmation_s
        if advisory:
            confirmed=confirmed and self.high_since is not None and timestamp_s-self.high_since >= self.high_confirmation_s
        if short_horizon and confirmed and not advisory:
            cap=min(measured_cap,self.max_speed_kmh)
            required=stopping_clearance(cap)
        self.previous_time=timestamp_s; self.previous_frame=lidar.get('sensor_frame')
        self.previous_observed_at=observed_at
        slowdown_only=(short_horizon or (level=='medium' or advisory)
            and clear and eligible and cadence_ok) and not confirmed
        if slowdown_only:
            cap=min(cap,10.)
            required=stopping_clearance(cap)
        disposition=('MUST_STOP' if hard or physical_stop or advisory and not confirmed and not slowdown_only
                     else 'CLEAR' if level == 'low' and not short_horizon and fresh and cadence_ok
                          and radar_ready and lidar.get('valid') is True and cap >= 5.
                     else 'CAUTIOUS_PROCEED' if (level == 'medium' or advisory or short_horizon) and (confirmed or slowdown_only)
                     else 'OBSERVATION_INSUFFICIENT')
        response_tier=('EMERGENCY_STOP' if emergency or hard and not physical_stop
            else 'BRAKE' if physical_stop else 'BOUNDED_SLOWDOWN' if slowdown_only
            else 'HOLD_FOR_OBSERVATION' if not confirmed and level != 'low'
            else 'LIMITED_HORIZON_BOUNDED' if short_horizon and cap > 10.
            else 'LIMITED_HORIZON_CRAWL' if short_horizon
            else 'HIGH_ADVISORY' if advisory else 'MEDIUM_ADVISORY' if level == 'medium' else 'NORMAL')
        result['behavior_policy']=dict(schema_version='risk_behavior/2.0', disposition=disposition,
            frame_id=frame_id, confirmed=bool(confirmed or slowdown_only or disposition == 'CLEAR'), physical_stop=bool(physical_stop),
            max_speed_kmh=cap if disposition in ('CAUTIOUS_PROCEED', 'CLEAR') else None,
            valid_until_s=(lidar['observed_at_s'] if finite(lidar.get('observed_at_s')) else timestamp_s)+.25,
            required_clearance_m=required, current_speed_required_clearance_m=current_required,
            lidar=lidar, clear_since_s=self.since,
            bounded_initial_slowdown=bool(slowdown_only), bounded_short_horizon=bool(short_horizon),
            confirmation_basis='fresh_measured_corridor' if slowdown_only else 'sustained_clearance',
            allow_positive_acceleration=not slowdown_only and not ttc_warning, absolute_stop_distance_m=.5,
            absolute_stop_active=any(finite(value) and value <= .5 for value in (distance,radar_distance)),
            classification_changed=False, policy_safety_credit=False,
            visual_advisory=advisory,high_clear_since_s=self.high_since,
            nominal_speed_cap_kmh=limit, response_tier=response_tier,
            raw_measured_speed_cap_kmh=raw_measured_cap, conservative_speed_cap_kmh=measured_cap,
            raw_physical_speed_cap_kmh=physical_cap,
            recent_physical_speed_cap_kmh=min((entry[1] for entry in self.cap_history),default=0.),
            cap_history_window_s=.5, cap_history_semantics='minimum recent measured bound; no clearance extrapolation',
            closing_ttc_s=ttc, emergency_evidence=bool(emergency),
            closing_ttc_response='BRAKE' if finite(ttc) and ttc<=1.5 else 'BOUNDED_DECELERATION' if ttc_warning else 'NONE',
            clearance_history_semantics='continuous fresh physical clearance, independent of visual class')
        if disposition == 'CAUTIOUS_PROCEED':
            result['unmodified_recommended_action']=risk.get('recommended_action')
            result['recommended_action']='monitor'
        return result


def potential_path_intrusion(observation, route, *, half_width_m, ego_front_m, speed_mps):
    """Screen short-horizon entrants using fresh measured track motion only."""
    if not observation or not route or len(route)<2:
        return False
    road=np.asarray(route,dtype=float)[:, :2]
    for entity in observation.get('entities',[]):
        age=entity.get('age_s');position=entity.get('position_xy_m');velocity=entity.get('velocity_xy_mps')
        if (not finite(age) or not 0<=age<=.25 or not isinstance(position,(list,tuple)) or len(position)!=2
                or not isinstance(velocity,(list,tuple)) or len(velocity)!=2
                or not all(finite(v) for v in (*position,*velocity))):
            continue
        deviation=entity.get('position_std_m')
        uncertainty=min(1.,2.*deviation) if finite(deviation) and deviation>=0 else 1.
        current_lateral=None
        current_tangent=None
        for horizon in (.0,.5,1.,1.5,2.):
            forecast_velocity = velocity
            if horizon>0 and entity.get('kind') == 'UNCLASSIFIED_RADAR_SURFACE':
                observed_velocity = entity.get('observed_velocity_xy_mps')
                if (entity.get('status') == 'TENTATIVE' or not finite(entity.get('motion_sample_count'))
                        or entity['motion_sample_count']<3
                        or not finite(entity.get('motion_observation_span_s'))
                        or entity['motion_observation_span_s']<.2
                        or not finite(entity.get('motion_position_residual_m'))
                        or entity['motion_position_residual_m']>.5
                        or not isinstance(observed_velocity,(list,tuple)) or len(observed_velocity)!=2
                        or not all(finite(v) for v in observed_velocity)):
                    continue
                forecast_velocity = observed_velocity
            point=np.asarray(position)+np.asarray(forecast_velocity)*horizon
            offset=0.;best=None
            for first,last in zip(road[:-1],road[1:]):
                delta=last-first;length=float(np.linalg.norm(delta))
                if length<.01:continue
                fraction=float(np.clip(((point-first)*delta).sum()/length**2,0.,1.))
                lateral=float(np.linalg.norm(point-first-fraction*delta))
                if best is None or lateral<best[0]:best=(lateral,offset+fraction*length,delta/length)
                offset+=length
            if horizon==0. and best is not None:
                current_lateral=best[0]
                current_tangent=best[2]
            # Uncertainty alone does not turn a stationary adjacent surface into
            # a crossing. A predicted entrant must move into the physical band.
            transverse_speed=(abs(float(current_tangent[0]*forecast_velocity[1]
                -current_tangent[1]*forecast_velocity[0])) if current_tangent is not None else 0.)
            # A same-lane stationary/longitudinal lead is handled by measured
            # distance and TTC, not by a fictitious future crossing at 6 km/h.
            enters=(best is not None and best[0]<=half_width_m+.35
                and (horizon==0. or current_lateral is not None and (
                    current_lateral<=half_width_m+.35 and transverse_speed>=.3
                    or current_lateral>half_width_m+.35
                    and current_lateral-best[0]>=max(.2,uncertainty))))
            if (enters
                      and ego_front_m+.4<best[1]<=ego_front_m+5.+max(speed_mps,6./3.6)*horizon):
                return True
    return False


def apply_behavior_decision(decision, risk, *, allow_restart=False):
    """Final authority: no timeout, liveness or network proposal may bypass it."""
    result=dict(decision); policy=risk.get('behavior_policy') or {}
    disposition=policy.get('disposition')
    stopped=result.get('action') in ('stop', 'emergency_brake') or result.get('emergency')
    permission=cautious_permission(risk, frame_id=result.get('frame_id'))
    source_action=result.get('source_step_action')
    from .task_feasibility import task_permission
    lane_motion=(result.get('action') in {'lane_change_left','lane_change_right'}
        and task_permission(risk,'CHANGE_LANE',frame_id=result.get('frame_id')))
    if disposition in ('MUST_STOP', 'OBSERVATION_INSUFFICIENT'):
        tier=policy.get('response_tier')
        explicit_emergency=source_action == 'EMERGENCY_BRAKE'
        emergency=(explicit_emergency or tier == 'EMERGENCY_STOP' or (tier is None and (
            risk.get('recommended_action') == 'emergency_brake' or risk.get('risk_level') == 'high'
            or result.get('emergency') or result.get('action') == 'emergency_brake')))
        result.update(action='emergency_brake' if emergency else 'stop', target_speed_kmh=0.,
            emergency=bool(emergency), target_lane=None, target_location=None,
            allow_positive_acceleration=False, longitudinal_control_mode='STOP',
            reason='risk_behavior_'+disposition.lower())
        result.pop('risk_motion_permission', None)
    elif (permission and not stopped and result.get('decision_status', 'READY') == 'READY'
          and not result.get('blocked_reason_codes')
          and (result.get('action') in ('accelerate','decelerate','keep_lane','turn_left','turn_right') or lane_motion)
          and (source_action in FORWARD_STEPS or source_action is None or lane_motion)):
        result.update(target_speed_kmh=min(float(result.get('target_speed_kmh', 0.)), permission['max_speed_kmh']),
                      risk_motion_permission=permission, longitudinal_control_mode='CAUTIOUS_SPEED_CAP')
        # Re-accelerating from rest to a bounded speed is not an unrestricted acceleration request.
        result['allow_positive_acceleration']=permission.get('allow_positive_acceleration',True) and (
            True if allow_restart else result.get('allow_positive_acceleration',True))
    elif (disposition == 'CAUTIOUS_PROCEED' or disposition == 'CLEAR'
          and policy.get('schema_version') == 'risk_behavior/2.0'
          and finite(policy.get('max_speed_kmh'))) and not permission:
        result.update(action='stop', target_speed_kmh=0., target_lane=None, target_location=None,
                      allow_positive_acceleration=False, reason='expired_cautious_motion_evidence')
    if disposition != 'CLEAR' or result.get('target_speed_kmh') != decision.get('target_speed_kmh'):
        for key in ('longitudinal_sequence_schema', 'target_acceleration_mps2', 'sequence_valid_until_s'):
            result.pop(key, None)
    return result, result != decision
