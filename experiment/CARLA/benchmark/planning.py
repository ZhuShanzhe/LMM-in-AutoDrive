"""Non-destructive task slicing plans; no implicit success or actor teleportation."""
from .catalog import ConfigError, number
from .task_oracle import load_profile
from .event_dependencies import audit_event_dependencies
from .capabilities import isolated_capability


def build_plan(catalog, selector, seed=0, pre_roll_m=150.0):
    pre_roll_m = number(pre_roll_m, 'pre_roll_m')
    if pre_roll_m < 0 or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError('nonnegative preparation distance and integer seed required')
    tasks = catalog.select(selector)
    isolated = selector != 'all'
    first = tasks[0]
    profiles, missing = {}, []
    for task in tasks:
        profile = load_profile(catalog, task)
        if profile is not None:
            profiles[task.task_id] = profile
            continue
        missing.append(task.task_id)
    event_audits={task_id:audit_event_dependencies(profile,catalog.events)
                  for task_id,profile in profiles.items()}
    selected_ids={t.task_id for t in tasks}
    prior_tasks={key:p['requires_task_success'] for key,p in profiles.items()
                 if p.get('requires_task_success')}
    external_dependencies=sorted({dep for deps in prior_tasks.values() for dep in deps
                                  if dep not in selected_ids})
    adapters={task.task_id:isolated_capability(catalog.scene_id,profiles.get(task.task_id)) for task in tasks}
    return dict(
        schema_version='task_execution_plan/1.0', scene_id=catalog.scene_id,
        source_file=catalog.source_file, source_sha256=catalog.source_sha256,
        map_name=catalog.map_name, mode='isolated' if isolated else 'full', seed=seed,
        execution_supported=False,
        isolated_adapters=adapters,
        preparation_start_m=max(0, first.announce_m - pre_roll_m) if isolated else 0,
        route_length_m=catalog.route_length_m,
        selected_task_ids=[t.task_id for t in tasks],
        model_commands=[dict(task_id=t.task_id, text=t.instruction,
                             announce_m=t.announce_m, activate_m=t.activate_m) for t in tasks],
        evaluation_windows=[dict(task_id=t.task_id, activate_m=t.activate_m, end_m=t.end_m)
                            for t in tasks],
        required_event_states={t.task_id: t.required_event_states for t in tasks
                               if t.required_event_states},
        # Preserve all event definitions for fixture review, including implicit links.
        fixture_review_events=list(catalog.events),
        oracle_profiles=profiles, missing_oracle_profiles=missing,
        event_dependency_audits=event_audits,
        required_task_success=prior_tasks,
        missing_selected_prerequisites=external_dependencies,
        blockers=['route_entry_not_audited', 'task_fixture_not_materialized']
                  + (['isolated_adapter_unavailable'] if isolated and not all(a['adapter_available'] for a in adapters.values()) else [])
                  + (['full_runner_invocation_required'] if not isolated else [])
                  + (['oracle_profiles_not_bound'] if missing else [])
                  + (['prior_task_evidence_required'] if external_dependencies else [])
                  + (['event_lifetime_conflict'] if any(not a['compatible'] for a in event_audits.values()) else []),
        history_equivalent_to_full_run=not isolated,
    )
