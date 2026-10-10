"""Bound diagnostic fragments without changing full-route execution."""


def remember_completed_plans(completed, selected, request_id, plan_status):
    result = set(completed)
    if plan_status == 'COMPLETED':
        result.update(identity for identity in selected if request_id.startswith(identity))
    return result


def fragment_tail_stop_reason(completed_at, now_s, progress_m):
    if completed_at is not None and now_s-completed_at[0] >= 3. and progress_m-completed_at[1] >= 100.:
        return 'parsed_plan_completed_with_diagnostic_tail'
    return None
