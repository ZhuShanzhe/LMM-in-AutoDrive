"""User-defined cyclist completion: predicted encounter window, no contact.

This event heuristic is not a measured rear-clearance/overtaking proof.
It does not grant motion authority or disable the physical safety controller.
"""
import math


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class CyclistEncounterWindow:
    def __init__(self, clearance_hold_s=2., prediction_horizon_s=45.):
        if not finite(clearance_hold_s) or clearance_hold_s <= 0:
            raise ValueError('positive encounter hold required')
        if not finite(prediction_horizon_s) or prediction_horizon_s <= 0:
            raise ValueError('positive prediction horizon required')
        self.hold_s = float(clearance_hold_s)
        self.horizon_s = float(prediction_horizon_s)
        self.events = {}

    @staticmethod
    def relevant(step):
        return (step.get('action') == 'PASS_BY'
                and (step.get('parameters') or {}).get('condition') == 'CYCLIST')

    def needs_scan(self, document):
        return any(self.relevant(step) and
                   (document['request_id'], step['step_id']) not in self.events
                   for step in document['intent']['steps'])

    def observe(self, document, tracks, *, frame_id, sensor_frame, timestamp_s,
                observed_at_s, progress_m, ego_speed_kmh, expected_speed_kmh,
                heading, contact_count):
        fresh = (finite(timestamp_s) and finite(observed_at_s)
                 and 0 <= timestamp_s-observed_at_s <= .25
                 and finite(progress_m) and finite(ego_speed_kmh)
                 and finite(expected_speed_kmh) and expected_speed_kmh > 0
                 and type(contact_count) is int and contact_count >= 0)
        result = {}
        for step in document['intent']['steps']:
            if not self.relevant(step):
                continue
            key = document['request_id'], step['step_id']
            event = self.events.get(key)
            if event is None and fresh:
                candidates = []
                for identity, track in tracks.items():
                    geometry = track.get('acquisition_geometry') or track
                    ahead, across = geometry.get('ahead_m'), geometry.get('lateral_m')
                    velocity = track.get('velocity')
                    if (track.get('category') != 'cyclist' or track.get('frame') != sensor_frame
                            or not track.get('velocity_valid') or velocity is None
                            or not finite(ahead) or not 0 < ahead <= 55
                            or not finite(across) or abs(across) > 4.5):
                        continue
                    target_speed = float(velocity[0]*heading[0]+velocity[1]*heading[1])
                    closing = expected_speed_kmh/3.6-target_speed
                    if not finite(closing) or closing <= .25:
                        continue
                    eta = ahead/closing
                    if eta <= self.horizon_s:
                        candidates.append((ahead, identity, eta, target_speed))
                if candidates:
                    ahead, identity, eta, target_speed = min(candidates)
                    event = dict(target_id=identity, started_at_s=timestamp_s,
                        entry_progress_m=progress_m, initial_ahead_m=ahead,
                        expected_speed_kmh=expected_speed_kmh,
                        observed_target_speed_mps=target_speed, predicted_encounter_s=eta,
                        finish_after_s=timestamp_s+eta+self.hold_s,
                        contact_baseline=contact_count, last_time_s=timestamp_s,
                        last_progress_m=progress_m, failed=False, completed_at_s=None)
                    self.events[key] = event
                    while len(self.events) > 128:
                        self.events.pop(next(iter(self.events)))
            if event is not None:
                delta = timestamp_s-event['last_time_s'] if finite(timestamp_s) else -1
                if delta < 0 or (fresh and contact_count < event['contact_baseline']):
                    event['failed'] = True
                if fresh and contact_count > event['contact_baseline']:
                    event['failed'] = True
                # Stopped time and missing observations cannot complete a passing event.
                moving = fresh and ego_speed_kmh >= 1.
                if delta > 0 and event['completed_at_s'] is None:
                    if not moving or delta > .25+1e-8:
                        event['finish_after_s'] += delta
                if finite(timestamp_s):
                    event['last_time_s'] = timestamp_s
                if fresh:
                    event['last_progress_m'] = progress_m
                if (fresh and moving and not event['failed']
                        and timestamp_s >= event['finish_after_s']
                        and progress_m-event['entry_progress_m'] >= 2.):
                    event['completed_at_s'] = event['completed_at_s'] or timestamp_s
            satisfied = bool(event and event['completed_at_s'] is not None
                             and not event['failed'] and fresh)
            fact = (dict(target_id=event['target_id'], category='cyclist',
                         source_request_id=key[0], source_step_id=key[1],
                         completed_at_s=event['completed_at_s'], event_status='COMPLETED',
                         completion_semantics='TIME_WINDOW_NO_CONTACT',
                         source='predicted_encounter_window_and_ego_contact_sensor')
                    if satisfied else None)
            reason = ('cyclist_encounter_contact_detected' if event and event['failed'] else
                      'cyclist_encounter_window_complete' if satisfied else
                      'cyclist_encounter_window_running' if event else
                      'cyclist_encounter_not_observed')
            result[step['step_id']] = dict(
                request_id=key[0], step_id=key[1], condition='CYCLIST', frame_id=frame_id,
                sensor_frame=sensor_frame, observed_at_s=observed_at_s,
                valid_until_s=observed_at_s+.25 if fresh else None,
                valid=fresh, satisfied=satisfied, can_proceed=satisfied,
                status='CLEAR' if satisfied else 'UNKNOWN', reason=reason,
                target_id=event['target_id'] if event else None, completed_event_fact=fact,
                encounter_window=dict(event) if event else None,
                contact_count=contact_count, completion_semantics='TIME_WINDOW_NO_CONTACT',
                measured_rear_clearance_required=False, continuous_target_tracking_required=False,
                source='cyclist_encounter_window', policy_truth_access=False)
        return result
