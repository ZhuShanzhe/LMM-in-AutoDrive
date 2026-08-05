"""Independent per-frame ground truth for CARLA evaluation.

The recorder deliberately consumes only simulator state, event contracts, and
event-runtime state.  It never reads semantic-model or controller outputs.
That separation is required before semantic alignment or control metrics can
be treated as evidence instead of self-evaluation.
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FRAME_GROUND_TRUTH_SCHEMA = "FrameGroundTruth/1.0.0"
TRUTH_QUALITY_VALUES = {
    "OBSERVED",
    "PARTIAL",
    "PROXY",
    "SCHEDULE_ONLY",
}
CONTROL_ACTIONS = {
    "keep_lane",
    "accelerate",
    "decelerate",
    "stop",
    "emergency_brake",
    "lane_change_left",
    "lane_change_right",
    "turn_left",
    "turn_right",
}


def _round(value: Any, digits: int = 3) -> float:
    return round(float(value), digits)


def _xyz(vector: Any) -> dict[str, float]:
    return {
        "x": _round(vector.x),
        "y": _round(vector.y),
        "z": _round(vector.z),
    }


def _speed_mps(velocity: Any) -> float:
    return math.sqrt(
        float(velocity.x) ** 2
        + float(velocity.y) ** 2
        + float(velocity.z) ** 2
    )


def _is_alive(actor: Any) -> bool:
    try:
        return bool(actor.is_alive)
    except (AttributeError, RuntimeError):
        return False


def _flatten_actor_bindings(
    actor_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for role, value in actor_bindings.items():
        if isinstance(value, (list, tuple)):
            for index, actor in enumerate(value, start=1):
                flattened[
                    "{0}_{1:03d}".format(role, index)
                ] = actor
        else:
            flattened[str(role)] = value
    return flattened


def _waypoint_for_actor(
    world: Any,
    actor: Any,
) -> Any | None:
    try:
        return world.get_map().get_waypoint(
            actor.get_location(),
            project_to_road=True,
        )
    except (AttributeError, RuntimeError, TypeError):
        return None


def _actor_snapshot(
    *,
    world: Any,
    actor: Any,
    role: str,
    ego_transform: Any,
    ego_velocity: Any,
) -> dict[str, Any] | None:
    if actor is None or not _is_alive(actor):
        return None
    try:
        transform = actor.get_transform()
        velocity = actor.get_velocity()
    except (AttributeError, RuntimeError):
        return None

    ego_location = ego_transform.location
    location = transform.location
    delta_x = float(location.x) - float(ego_location.x)
    delta_y = float(location.y) - float(ego_location.y)
    delta_z = float(location.z) - float(ego_location.z)
    yaw_rad = math.radians(
        float(ego_transform.rotation.yaw)
    )
    forward_x = math.cos(yaw_rad)
    forward_y = math.sin(yaw_rad)
    right_x = -forward_y
    right_y = forward_x
    longitudinal_m = (
        delta_x * forward_x + delta_y * forward_y
    )
    lateral_m = delta_x * right_x + delta_y * right_y
    distance_m = math.sqrt(
        delta_x * delta_x
        + delta_y * delta_y
        + delta_z * delta_z
    )
    ego_forward_speed = (
        float(ego_velocity.x) * forward_x
        + float(ego_velocity.y) * forward_y
    )
    actor_forward_speed = (
        float(velocity.x) * forward_x
        + float(velocity.y) * forward_y
    )
    closing_speed_mps = (
        ego_forward_speed - actor_forward_speed
    )
    time_to_collision_s = None
    if longitudinal_m > 0.0 and closing_speed_mps > 0.1:
        time_to_collision_s = (
            longitudinal_m / closing_speed_mps
        )

    waypoint = _waypoint_for_actor(world, actor)
    attributes = getattr(actor, "attributes", {}) or {}
    return {
        "role": role,
        "actor_id": int(actor.id),
        "type_id": str(
            getattr(actor, "type_id", "unknown")
        ),
        "blueprint_role_name": attributes.get("role_name"),
        "alive": True,
        "transform": {
            "location": _xyz(location),
            "rotation": {
                "pitch": _round(transform.rotation.pitch),
                "yaw": _round(transform.rotation.yaw),
                "roll": _round(transform.rotation.roll),
            },
        },
        "velocity_mps": _xyz(velocity),
        "speed_mps": _round(_speed_mps(velocity)),
        "lane": {
            "road_id": (
                int(waypoint.road_id)
                if waypoint is not None
                else None
            ),
            "section_id": (
                int(waypoint.section_id)
                if waypoint is not None
                else None
            ),
            "lane_id": (
                int(waypoint.lane_id)
                if waypoint is not None
                else None
            ),
            "is_junction": (
                bool(waypoint.is_junction)
                if waypoint is not None
                else False
            ),
        },
        "relation_to_ego": {
            "longitudinal_m": _round(longitudinal_m),
            "lateral_m": _round(lateral_m),
            "euclidean_distance_m": _round(distance_m),
            "relative_closing_speed_mps": _round(
                closing_speed_mps
            ),
            "time_to_collision_s": (
                _round(time_to_collision_s)
                if time_to_collision_s is not None
                else None
            ),
            "position": (
                "AHEAD"
                if longitudinal_m >= 0.0
                else "BEHIND"
            ),
        },
    }


def _ego_snapshot(
    world: Any,
    ego: Any,
) -> dict[str, Any]:
    transform = ego.get_transform()
    velocity = ego.get_velocity()
    waypoint = _waypoint_for_actor(world, ego)
    return {
        "actor_id": int(ego.id),
        "transform": {
            "location": _xyz(transform.location),
            "rotation": {
                "pitch": _round(transform.rotation.pitch),
                "yaw": _round(transform.rotation.yaw),
                "roll": _round(transform.rotation.roll),
            },
        },
        "velocity_mps": _xyz(velocity),
        "speed_kmh": _round(
            3.6 * _speed_mps(velocity)
        ),
        "lane": {
            "road_id": (
                int(waypoint.road_id)
                if waypoint is not None
                else None
            ),
            "section_id": (
                int(waypoint.section_id)
                if waypoint is not None
                else None
            ),
            "lane_id": (
                int(waypoint.lane_id)
                if waypoint is not None
                else None
            ),
            "is_junction": (
                bool(waypoint.is_junction)
                if waypoint is not None
                else False
            ),
        },
    }


def _requirements(
    event: Mapping[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    contract = event.get("ground_truth", {})
    return (
        [
            str(role)
            for role in contract.get("actor_roles", [])
        ],
        [
            str(prefix)
            for prefix in contract.get(
                "actor_role_prefixes",
                [],
            )
        ],
        [
            str(flag)
            for flag in contract.get(
                "required_runtime_flags",
                [],
            )
        ],
    )


def _binding_evidence(
    event: Mapping[str, Any],
    snapshots: Mapping[str, Mapping[str, Any]],
    runtime_state: Mapping[str, Any],
) -> dict[str, Any]:
    roles, prefixes, flags = _requirements(event)
    contract = event.get("ground_truth", {})
    max_actor_distance_m = contract.get("max_actor_distance_m")
    max_actor_lateral_m = contract.get("max_actor_lateral_m")
    observed: list[str] = []
    missing: list[str] = []
    out_of_range: list[str] = []

    def actor_is_in_range(role: str) -> bool:
        snapshot = snapshots.get(role)
        if snapshot is None:
            return False
        relation = snapshot.get("relation_to_ego", {})
        distance = relation.get("euclidean_distance_m")
        lateral = relation.get("lateral_m")
        if (
            max_actor_distance_m is not None
            and distance is not None
            and float(distance) > float(max_actor_distance_m)
        ):
            out_of_range.append(role)
            return False
        if (
            max_actor_lateral_m is not None
            and lateral is not None
            and abs(float(lateral)) > float(max_actor_lateral_m)
        ):
            out_of_range.append(role)
            return False
        return True

    for role in roles:
        if actor_is_in_range(role):
            observed.append(role)
        else:
            missing.append(role)
    for prefix in prefixes:
        matches = sorted(
            role
            for role in snapshots
            if role.startswith(prefix) and actor_is_in_range(role)
        )
        if matches:
            observed.extend(matches)
        else:
            missing.append(prefix + "*")
    for flag in flags:
        if runtime_state.get(flag) is True:
            observed.append("runtime:" + flag)
        else:
            missing.append("runtime:" + flag)
    return {
        "required": roles
        + [prefix + "*" for prefix in prefixes]
        + ["runtime:" + flag for flag in flags],
        "observed": sorted(set(observed)),
        "missing": sorted(set(missing)),
        "out_of_range": sorted(set(out_of_range)),
    }


def _truth_quality(
    event: Mapping[str, Any],
    evidence: Mapping[str, Sequence[str]],
) -> str:
    mode = str(
        event.get("ground_truth", {}).get(
            "evidence_mode",
            "schedule_only",
        )
    ).lower()
    required = list(evidence["required"])
    missing = list(evidence["missing"])
    if mode == "schedule_only":
        return "SCHEDULE_ONLY"
    if mode == "proxy":
        return "PROXY" if not missing else "SCHEDULE_ONLY"
    if mode != "observed":
        raise ValueError(
            "unsupported ground-truth evidence_mode: "
            + mode
        )
    if not required:
        return "OBSERVED"
    if not missing:
        return "OBSERVED"
    if len(missing) < len(required):
        return "PARTIAL"
    return "SCHEDULE_ONLY"


def _conditional_contract(
    contract: Mapping[str, Any],
    runtime_state: Mapping[str, Any],
) -> dict[str, Any]:
    result = {
        "risk_labels": list(
            contract.get("risk_labels", [])
        ),
        "allowed_control_actions": list(
            contract.get(
                "allowed_control_actions",
                [],
            )
        ),
        "forbidden_control_actions": list(
            contract.get(
                "forbidden_control_actions",
                [],
            )
        ),
    }
    conditional = contract.get("conditional")
    if not isinstance(conditional, Mapping):
        return result
    flag = str(conditional.get("runtime_flag", ""))
    branch = (
        "when_true"
        if runtime_state.get(flag) is True
        else "when_false"
    )
    override = conditional.get(branch, {})
    if isinstance(override, Mapping):
        for name in (
            "risk_labels",
            "allowed_control_actions",
            "forbidden_control_actions",
        ):
            if name in override:
                result[name] = list(override[name])
    return result


def validate_event_ground_truth_contracts(
    events: Sequence[Mapping[str, Any]],
) -> None:
    """Reject incomplete or internally contradictory truth contracts."""

    for event in events:
        event_id = str(event.get("id", ""))
        contract = event.get("ground_truth")
        if not isinstance(contract, Mapping):
            raise ValueError(
                event_id
                + " requires a ground_truth contract"
            )
        mode = str(
            contract.get("evidence_mode", "")
        ).lower()
        if mode not in {
            "observed",
            "proxy",
            "schedule_only",
        }:
            raise ValueError(
                event_id
                + " has invalid ground_truth evidence_mode"
            )
        risks = contract.get("risk_labels")
        if not isinstance(risks, list) or not risks:
            raise ValueError(
                event_id
                + " requires non-empty ground_truth risk_labels"
            )
        allowed = {
            str(action).lower()
            for action in contract.get(
                "allowed_control_actions",
                [],
            )
        }
        forbidden = {
            str(action).lower()
            for action in contract.get(
                "forbidden_control_actions",
                [],
            )
        }
        if not allowed:
            raise ValueError(
                event_id
                + " requires allowed_control_actions"
            )
        unsupported = (
            allowed | forbidden
        ) - CONTROL_ACTIONS
        if unsupported:
            raise ValueError(
                event_id
                + " contains unsupported control actions: "
                + ", ".join(sorted(unsupported))
            )
        if allowed & forbidden:
            raise ValueError(
                event_id
                + " allows and forbids the same action: "
                + ", ".join(
                    sorted(allowed & forbidden)
                )
            )
        roles, prefixes, flags = _requirements(event)
        if (
            mode in {"observed", "proxy"}
            and not (roles or prefixes or flags)
        ):
            raise ValueError(
                event_id
                + " requires actor or runtime evidence"
            )
        conditional = contract.get("conditional")
        if conditional is None:
            continue
        if not isinstance(conditional, Mapping):
            raise ValueError(
                event_id
                + " ground_truth.conditional must be an object"
            )
        if not conditional.get("runtime_flag"):
            raise ValueError(
                event_id
                + " conditional runtime_flag is required"
            )
        for branch in ("when_false", "when_true"):
            if not isinstance(
                conditional.get(branch),
                Mapping,
            ):
                raise ValueError(
                    event_id
                    + " conditional "
                    + branch
                    + " is required"
                )


class FrameGroundTruthRecorder:
    """Write independent, exact-frame CARLA truth records."""

    def __init__(
        self,
        path: Path,
        *,
        scene_id: str,
        events: Sequence[Mapping[str, Any]],
        every_n_frames: int = 1,
    ) -> None:
        if every_n_frames < 1:
            raise ValueError(
                "every_n_frames must be at least 1"
            )
        self.path = Path(path)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self._stream = self.path.open(
            "w",
            encoding="utf-8",
        )
        self.scene_id = str(scene_id)
        self.events = [
            dict(event)
            for event in events
        ]
        self.every_n_frames = int(every_n_frames)
        self.record_count = 0
        self._seen_frames: set[int] = set()
        self._quality_counts: Counter[str] = Counter()
        self._event_observed_frames: Counter[str] = Counter()
        self._closed = False

    def record(
        self,
        *,
        world: Any,
        ego: Any,
        simulation_frame: int,
        timestamp_s: float,
        route_s_m: float,
        event_states: Mapping[str, str],
        actor_bindings: Mapping[str, Any],
        runtime_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        frame = int(simulation_frame)
        if frame % self.every_n_frames != 0:
            return None
        if frame in self._seen_frames:
            raise ValueError(
                "duplicate ground-truth simulation_frame: "
                + str(frame)
            )
        runtime = dict(runtime_state or {})
        ego_transform = ego.get_transform()
        ego_velocity = ego.get_velocity()
        active_event_definitions = [
            event
            for event in self.events
            if event_states.get(
                str(event["id"])
            )
            == "ACTIVE"
        ]
        required_roles: set[str] = set()
        required_prefixes: set[str] = set()
        for event in active_event_definitions:
            roles, prefixes, _ = _requirements(
                event
            )
            required_roles.update(roles)
            required_prefixes.update(prefixes)

        flattened = _flatten_actor_bindings(
            actor_bindings
        )
        flattened = {
            role: actor
            for role, actor in flattened.items()
            if (
                role in required_roles
                or any(
                    role.startswith(prefix)
                    for prefix in required_prefixes
                )
            )
        }
        snapshots: dict[str, dict[str, Any]] = {}
        unavailable_roles: list[str] = []
        for role, actor in sorted(flattened.items()):
            snapshot = _actor_snapshot(
                world=world,
                actor=actor,
                role=role,
                ego_transform=ego_transform,
                ego_velocity=ego_velocity,
            )
            if snapshot is None:
                unavailable_roles.append(role)
            else:
                snapshots[role] = snapshot

        active_events: list[dict[str, Any]] = []
        for event in active_event_definitions:
            event_id = str(event["id"])
            evidence = _binding_evidence(
                event,
                snapshots,
                runtime,
            )
            quality = _truth_quality(
                event,
                evidence,
            )
            contract = _conditional_contract(
                event.get("ground_truth", {}),
                runtime,
            )
            event_truth = {
                "event_id": event_id,
                "scenario": event.get(
                    "scenario",
                    event.get("completion", event_id),
                ),
                "state": "ACTIVE",
                "runtime_phase": runtime.get(
                    str(
                        event.get(
                            "ground_truth",
                            {},
                        ).get(
                            "runtime_phase_key",
                            "",
                        )
                    )
                ),
                "risk_labels": contract["risk_labels"],
                "expected_control": {
                    "allowed_actions": contract[
                        "allowed_control_actions"
                    ],
                    "forbidden_actions": contract[
                        "forbidden_control_actions"
                    ],
                },
                "truth_quality": quality,
                "evidence": evidence,
            }
            active_events.append(event_truth)
            if quality == "OBSERVED":
                self._event_observed_frames[
                    event_id
                ] += 1

        if not active_events:
            frame_quality = "SCHEDULE_ONLY"
        elif all(
            event["truth_quality"] == "OBSERVED"
            for event in active_events
        ):
            frame_quality = "OBSERVED"
        elif any(
            event["truth_quality"] == "OBSERVED"
            for event in active_events
        ):
            frame_quality = "PARTIAL"
        elif any(
            event["truth_quality"] == "PROXY"
            for event in active_events
        ):
            frame_quality = "PROXY"
        elif any(
            event["truth_quality"] == "PARTIAL"
            for event in active_events
        ):
            frame_quality = "PARTIAL"
        else:
            frame_quality = "SCHEDULE_ONLY"

        record = {
            "schema_version": FRAME_GROUND_TRUTH_SCHEMA,
            "scene_id": self.scene_id,
            "simulation_frame": frame,
            "timestamp_s": _round(timestamp_s),
            "route_s_m": _round(route_s_m),
            "ego": _ego_snapshot(world, ego),
            "event_states": {
                str(key): str(value)
                for key, value in event_states.items()
            },
            "active_events": active_events,
            "actors": snapshots,
            "unavailable_actor_roles": sorted(
                unavailable_roles
            ),
            "runtime_state": runtime,
            "frame_truth_quality": frame_quality,
            "claim_eligible": (
                bool(active_events)
                and all(
                    event["truth_quality"] == "OBSERVED"
                    for event in active_events
                )
            ),
            "provenance": {
                "source": "CARLA_SIMULATOR_STATE",
                "event_phase_source": "ROUTE_SCHEDULE",
                "model_output_used": False,
                "adjacent_frame_fill_used": False,
                "synchronization_key": "simulation_frame",
            },
        }
        self._stream.write(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
        self._stream.flush()
        self._seen_frames.add(frame)
        self.record_count += 1
        self._quality_counts[frame_quality] += 1
        return record

    def summary(self) -> dict[str, Any]:
        event_evidence_modes = {
            str(event["id"]): str(
                event["ground_truth"][
                    "evidence_mode"
                ]
            ).upper()
            for event in self.events
        }
        return {
            "schema_version": FRAME_GROUND_TRUTH_SCHEMA,
            "path": str(self.path),
            "records": self.record_count,
            "frame_quality_counts": {
                quality: self._quality_counts[quality]
                for quality in sorted(
                    TRUTH_QUALITY_VALUES
                )
            },
            "observed_event_frames": dict(
                sorted(
                    self._event_observed_frames.items()
                )
            ),
            "event_evidence_modes": (
                event_evidence_modes
            ),
            "claim_capable_event_contracts": sorted(
                event_id
                for event_id, mode in (
                    event_evidence_modes.items()
                )
                if mode == "OBSERVED"
            ),
            "model_output_used": False,
        }

    def close(self) -> None:
        if not self._closed:
            self._stream.close()
            self._closed = True


def observed_event_ids(
    record: Mapping[str, Any],
) -> set[str]:
    """Return active event IDs eligible for measured claims."""

    return {
        str(event["event_id"])
        for event in record.get("active_events", [])
        if event.get("truth_quality") == "OBSERVED"
    }


def observed_risk_labels(
    record: Mapping[str, Any],
) -> set[str]:
    labels: set[str] = set()
    for event in record.get("active_events", []):
        if event.get("truth_quality") != "OBSERVED":
            continue
        labels.update(
            str(label)
            for label in event.get("risk_labels", [])
        )
    return labels


def observed_control_contract(
    record: Mapping[str, Any],
) -> tuple[set[str], set[str]]:
    allowed: set[str] = set()
    forbidden: set[str] = set()
    for event in record.get("active_events", []):
        if event.get("truth_quality") != "OBSERVED":
            continue
        control = event.get("expected_control", {})
        allowed.update(
            str(action).lower()
            for action in control.get(
                "allowed_actions",
                [],
            )
        )
        forbidden.update(
            str(action).lower()
            for action in control.get(
                "forbidden_actions",
                [],
            )
        )
    return allowed, forbidden


def unique_frames(
    records: Iterable[Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for raw in records:
        record = dict(raw)
        frame = int(record["simulation_frame"])
        if frame in result:
            raise ValueError(
                "duplicate simulation_frame: "
                + str(frame)
            )
        result[frame] = record
    return result
