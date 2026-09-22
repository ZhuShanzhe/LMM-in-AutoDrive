"""Select assessment targets while retaining independent prerequisite evidence."""
from .catalog import ConfigError
from .task_oracle import load_profile


def validate_assessment_args(args, scene):
    if args.benchmark_task != 'all':
        if not args.benchmark_assessment:
            raise ConfigError('--benchmark-task requires --benchmark-assessment')
        from .catalog import load_catalog
        assessment_selection(load_catalog(scene), args.benchmark_task)


def assessment_selection(catalog, selector='all'):
    selected = {task.task_id for task in catalog.select(selector)}
    by_id = {task.task_id: task for task in catalog.tasks}
    included = set()
    visiting = set()
    def visit(identity):
        if identity in visiting:
            raise ConfigError('cyclic assessment prerequisite: ' + identity)
        if identity in included:
            return
        if identity not in by_id:
            raise ConfigError('unknown assessment prerequisite: ' + identity)
        visiting.add(identity)
        profile = load_profile(catalog, by_id[identity])
        for dependency in (profile or {}).get('requires_task_success', []):
            visit(dependency)
        visiting.remove(identity)
        included.add(identity)
    for identity in selected:
        visit(identity)
    return ([task for task in catalog.tasks if task.task_id in selected],
            [task for task in catalog.tasks if task.task_id in included])
