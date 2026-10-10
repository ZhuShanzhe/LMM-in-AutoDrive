"""Bind instruction steps to planner-owned geometry, never scenario truth."""
from bisect import bisect_left, bisect_right
from copy import deepcopy
import math

from benchmark.route_audit import audit_route


class RoadTargetResolver:
    def __init__(self, route):
        self.route = deepcopy(route)
        audit = audit_route(self.route)
        if audit['suspicious_gaps']:
            raise ValueError('road_target_route_discontinuous')
        self.visits = audit['junction_visits']
        self.ends = [visit['end_m'] for visit in self.visits]
        self.distances = [point['distance_m'] for point in self.route]
        self.request_id = None
        self.pinned = {}

    def resolve(self, document, *, frame_id, progress_m):
        if not math.isfinite(progress_m):
            raise ValueError('Finite route progress required')
        identity = document['request_id']
        if self.request_id != identity:
            self.request_id = identity
            self.pinned.clear()
        result = {}
        for step in document['intent']['steps']:
            if step.get('action') != 'TURN':
                continue
            direction = str((step.get('parameters') or {}).get('direction', '')).upper()
            item = dict(request_id=identity, step_id=step['step_id'], frame_id=frame_id,
                source='planned_route_topology', valid=False, reachable=False,
                ambiguous=False, requested_direction=direction)
            visit = self.pinned.get(step['step_id'])
            if visit is None:
                index = bisect_right(self.ends, progress_m)
                visit = self.visits[index] if index < len(self.visits) else None
            if visit is None:
                item['reason'] = 'road_target_no_next_junction'
            elif not visit['complete']:
                item['reason'] = 'road_target_junction_incomplete'
            elif (not visit['distinct_road_turn_candidate']
                    or visit['direction'] != direction):
                item.update(reason='road_target_direction_conflict',
                    observed_direction=visit['direction'], junction=deepcopy(visit))
            else:
                index = min(bisect_left(self.distances, visit['end_m']), len(self.route)-1)
                target = {key: float(self.route[index][key]) for key in ('x', 'y', 'z', 'yaw')}
                item.update(valid=True, reachable=True, reason='road_target_bound',
                    target_location=target, junction=deepcopy(visit),
                    legality_scope='planner_geometry_not_signal_or_obstacle_clearance')
            result[step['step_id']] = item
        return result

    def pin(self, step, evidence, *, frame_id):
        """Latch only the step selected by the executor, not future steps."""
        if (step.get('action') == 'TURN' and isinstance(evidence, dict)
                and evidence.get('request_id') == self.request_id
                and evidence.get('step_id') == step.get('step_id')
                and evidence.get('frame_id') == frame_id
                and evidence.get('source') == 'planned_route_topology'
                and evidence.get('valid') is True and evidence.get('reachable') is True
                and evidence.get('requested_direction') == (step.get('parameters') or {}).get('direction')
                and evidence.get('junction') in self.visits):
            self.pinned.setdefault(step['step_id'], deepcopy(evidence['junction']))
