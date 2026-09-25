"""Read legacy scene configurations without importing simulator or model code."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

CONFIG_ROOT = Path(__file__).resolve().parents[1] / 'configs'
SCENES = {
    'scene_1': ('basic_voice_urban_5km.json', 5000.0),
    'scene_2': ('scene_2_town05_runtime.json', 8000.0),
    'scene_3': ('scene_3_emergency_6km_runtime.json', 6000.0),
}


class ConfigError(ValueError):
    pass


def source_fingerprint(payload: bytes) -> str:
    """Bind scene content while ignoring only Git's CRLF checkout conversion."""
    return hashlib.sha256(payload.replace(b'\r\n', b'\n')).hexdigest()


def number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f'{label}: expected a finite number')
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f'{label}: expected a finite number')
    return result


def records(value: Any, label: str) -> list[dict]:
    if not isinstance(value, list) or any(not isinstance(x, dict) for x in value):
        raise ConfigError(f'{label}: expected a list of objects')
    seen = set()
    for item in value:
        key = item.get('id')
        if not isinstance(key, str) or not key.strip() or key in seen:
            raise ConfigError(f'{label}: missing or duplicate id: {key!r}')
        seen.add(key)
    return value


@dataclass(frozen=True)
class Task:
    task_id: str
    order: int
    source_order: int
    instruction: str
    announce_m: float
    activate_m: float
    end_m: float | None
    required_event_states: dict
    linked_event_ids: tuple[str, ...]
    source_command: dict


@dataclass(frozen=True)
class Catalog:
    schema_version: str
    scene_id: str
    source_scene_id: str
    source_file: str
    source_sha256: str
    map_name: str
    route_length_m: float
    tasks: tuple[Task, ...]
    events: tuple[dict, ...]

    def to_dict(self) -> dict:
        return asdict(self)

    def select(self, selector: str) -> tuple[Task, ...]:
        if selector == 'all':
            return self.tasks
        for task in self.tasks:
            if selector == task.task_id or selector == str(task.order):
                return (task,)
        raise ConfigError(f'unknown task {selector!r} in {self.scene_id}')


def load_catalog(scene: str, config_root: Path = CONFIG_ROOT) -> Catalog:
    if scene not in SCENES:
        raise ConfigError(f'unknown scene: {scene}')
    path = config_root / SCENES[scene][0]
    payload = path.read_bytes()
    try:
        raw = json.loads(payload)
    except (ValueError, UnicodeError) as error:
        raise ConfigError(f'{path.name}: invalid JSON') from error
    if not isinstance(raw, dict):
        raise ConfigError('scene configuration must be an object')
    map_value = raw.get('map')
    map_name = map_value.get('name') if isinstance(map_value, dict) else map_value
    if not isinstance(map_name, str) or not map_name:
        raise ConfigError('map name is missing')
    route = raw.get('route', map_value if isinstance(map_value, dict) else {})
    if not isinstance(route, dict):
        raise ConfigError('route must be an object')
    length = number(route.get('target_length_m', route.get('length_m')), 'route length')
    if length <= 0:
        raise ConfigError('route length must be positive')
    commands = raw.get('commands') if scene != 'scene_3' else raw.get('voice_input', {}).get('commands')
    commands = records(commands, 'commands')
    if not commands:
        raise ConfigError('commands must not be empty')
    events = records(raw.get('special_events', raw.get('events', [])), 'events')
    event_ids = {e['id'] for e in events}
    command_ids = {c['id'] for c in commands}
    for event in events:
        reference = event.get('voice_command_id')
        if reference is not None and reference not in command_ids:
            raise ConfigError(f"{event['id']}: unknown voice command {reference}")
    tasks = []
    for index, command in enumerate(commands, 1):
        key = command['id']
        announce = number(command.get('announce_at_m', command.get('trigger_progress_m')), f'{key}.announce')
        activate = number(command.get('activate_at_m', announce), f'{key}.activate')
        end = command.get('end_progress_m')
        end = number(end, f'{key}.end') if end is not None else None
        if not 0 <= announce <= activate < length or (end is not None and not activate < end <= length):
            raise ConfigError(f'{key}: invalid task distance bounds')
        text = command.get('voice_text', command.get('text'))
        if not isinstance(text, str) or not text.strip():
            raise ConfigError(f'{key}: instruction text is missing')
        dependencies = command.get('requires_event_states', {})
        if not isinstance(dependencies, dict) or any(
            ref not in event_ids or not isinstance(state, str) or not state
            for ref, state in dependencies.items()
        ):
            raise ConfigError(f'{key}: invalid event dependency')
        linked = tuple(e['id'] for e in events if e.get('voice_command_id') == key)
        tasks.append(Task(key, 0, index, text, announce, activate, end, dependencies, linked, command))
    tasks.sort(key=lambda t: (t.activate_m, t.source_order))
    tasks = tuple(Task(**{**asdict(t), 'order': i}) for i, t in enumerate(tasks, 1))
    source_hash = source_fingerprint(payload)
    return Catalog('benchmark_catalog/1.0', scene, raw.get('scene_id', raw.get('scenario_id', scene)),
                   path.name, source_hash, map_name, length, tasks, tuple(events))


def validate_catalog(catalog: Catalog) -> dict:
    from .task_oracle import load_profile
    issues = []

    def issue(code: str, message: str, task_id: str | None = None, severity: str = 'warning'):
        issues.append(dict(code=code, severity=severity, task_id=task_id, message=message))

    if catalog.route_length_m < SCENES[catalog.scene_id][1]:
        issue('route_too_short', 'Configured route is shorter than the scene requirement.', severity='error')
    if any(t.source_order != t.order for t in catalog.tasks):
        issue('source_order_differs', 'Task numbers follow activation distance; source IDs remain unchanged.')
    for task in catalog.tasks:
        if task.end_m is None:
            issue('missing_task_end', 'No explicit task end window; next command is not a completion criterion.', task.task_id)
        if task.required_event_states or task.linked_event_ids:
            issue('fixture_required', 'Isolated execution requires event actors and prerequisite state setup.', task.task_id)
        if load_profile(catalog, task) is None:
            issue('oracle_not_registered', 'No matching independent criterion profile.', task.task_id)
        else:
            issue('oracle_runtime_not_connected', 'Criterion profile exists; runtime truth and fixture remain unverified.', task.task_id)
    for i, task in enumerate(catalog.tasks):
        if task.end_m is not None:
            for other in catalog.tasks[i + 1:]:
                if other.activate_m < task.end_m:
                    issue('overlapping_tasks', f'Execution window overlaps {other.task_id}; preserve explicit concurrency.', task.task_id)
    issue('route_geometry_unverified', 'Configured distance does not verify road topology, lane count or legal turns.')
    return {
        'scene_id': catalog.scene_id,
        'config_valid': not any(x['severity'] == 'error' for x in issues),
        'benchmark_ready': False,
        'issues': issues,
    }
