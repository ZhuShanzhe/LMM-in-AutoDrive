"""One registry for implemented isolated runners, distinct from verified readiness."""


def isolated_capability(scene,profile):
    result=dict(adapter_available=False,kind=None,model_inference=False,
        physics_verified=False,scope='derived_simulator_acceptance_fixture',
        requirements=['matching_route_file','enabled_background_traffic_config',
                      'idle_carla_0.9.16_server','geometry_and_actor_preflight'],
        reason=None)
    if profile is None:
        result['reason']='missing_bound_profile'
        return result
    if profile.get('requires_task_success') or profile.get('constraints'):
        result['reason']='prior_task_or_interval_evidence_not_reconstructed'
        return result
    named={('scene_2','s2_t05_cmd_03'):'composite',
           ('scene_2','s2_t05_cmd_07'):'overtake',
           ('scene_3','scene3_worker_crossing'):'pedestrian'}
    kind=named.get((scene,profile['task_id']))
    steps=profile['steps']
    if kind is None and len(steps)==1 and steps[0]['kind'] in {'speed','lane_change','turn'}:
        kind=steps[0]['kind']
    result.update(adapter_available=kind is not None,kind=kind,
                  reason=None if kind else 'no_isolated_adapter_for_step_sequence')
    return result
