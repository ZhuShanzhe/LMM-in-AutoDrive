"""Create frame-aligned Qwen manifests from recorded CARLA keyframes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from scene_understanding.core.validate_scene_output import ENUMS
from scene_understanding.core.world_state import validate_world_state


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    return records


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def _resolved_path(value: Any, *, base_dir: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}: expected a non-empty path")
    path = Path(value)
    return (base_dir / path).resolve() if not path.is_absolute() else path.resolve()


def _render_prompt(template: str, *, frame_id: str, camera_name: str) -> str:
    rendered = template
    for key, value in {
        "frame_id": frame_id,
        "source": "carla",
        "camera_name": camera_name,
    }.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def _valid_bbox(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(
            not isinstance(item, bool) and isinstance(item, (int, float)) and 0 <= item <= 1
            for item in value
        )
        and value[0] < value[2]
        and value[1] < value[3]
    )


def validate_projection_record(
    projection: Any,
    *,
    frame_id: str,
    camera_name: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(projection, dict):
        return ["projection: expected an object"]
    if projection.get("schema_version") != "1.0":
        errors.append("projection.schema_version: expected '1.0'")
    if projection.get("frame_id") != frame_id:
        errors.append("projection.frame_id: must match WorldState")
    if projection.get("camera_name") != camera_name:
        errors.append("projection.camera_name: must match capture record")
    objects = projection.get("objects")
    if not isinstance(objects, list):
        errors.append("projection.objects: expected an array")
        return errors
    object_ids: set[str] = set()
    for index, obj in enumerate(objects):
        path = f"projection.objects[{index}]"
        if not isinstance(obj, dict):
            errors.append(f"{path}: expected an object")
            continue
        object_id = obj.get("world_object_id")
        if not isinstance(object_id, str) or not object_id:
            errors.append(f"{path}.world_object_id: expected a non-empty string")
        elif object_id in object_ids:
            errors.append(f"{path}.world_object_id: duplicate {object_id}")
        else:
            object_ids.add(object_id)
        if obj.get("category") not in ENUMS["category"]:
            errors.append(f"{path}.category: unsupported category")
        if not _valid_bbox(obj.get("bbox_2d")):
            errors.append(f"{path}.bbox_2d: expected normalized xyxy coordinates")
    return errors


def write_capture_bundle(
    output_dir: Path,
    *,
    camera_record: Mapping[str, Any],
    world_state: dict[str, Any],
    projection_record: dict[str, Any],
) -> dict[str, Any]:
    """Persist one synchronized capture and return its index record."""

    world_errors = validate_world_state(world_state)
    if world_errors:
        raise ValueError("invalid WorldState: " + "; ".join(world_errors))
    frame_id = str(world_state["frame_id"])
    simulation_frame = int(world_state["simulation_frame"])
    if int(camera_record.get("frame", -1)) != simulation_frame:
        raise ValueError("camera frame and WorldState simulation_frame must match")
    camera_name = str(camera_record.get("camera_name", ""))
    if not camera_name:
        raise ValueError("camera_record.camera_name is required")
    projection_errors = validate_projection_record(
        projection_record,
        frame_id=frame_id,
        camera_name=camera_name,
    )
    if projection_errors:
        raise ValueError("invalid projection record: " + "; ".join(projection_errors))

    image_path = Path(str(camera_record.get("image_path", ""))).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"camera image not found: {image_path}")
    world_path = output_dir / "world_states" / f"{frame_id}.json"
    projection_path = output_dir / "projections" / f"{frame_id}.json"
    world_path.parent.mkdir(parents=True, exist_ok=True)
    projection_path.parent.mkdir(parents=True, exist_ok=True)
    world_path.write_text(
        json.dumps(world_state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    projection_path.write_text(
        json.dumps(projection_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "frame_id": frame_id,
        "simulation_frame": simulation_frame,
        "timestamp_s": float(world_state["timestamp_s"]),
        "camera_name": camera_name,
        "image_path": str(image_path),
        "world_state_path": str(world_path.resolve()),
        "projection_path": str(projection_path.resolve()),
    }


def build_manifest_record(
    capture: Mapping[str, Any],
    *,
    base_dir: Path,
    prompt_template: str,
    require_image: bool = True,
) -> dict[str, Any]:
    frame_id = str(capture.get("frame_id", ""))
    camera_name = str(capture.get("camera_name", ""))
    if not frame_id or not camera_name:
        raise ValueError("capture frame_id and camera_name are required")
    image_path = _resolved_path(capture.get("image_path"), base_dir=base_dir, field="image_path")
    if require_image and not image_path.is_file():
        raise FileNotFoundError(f"image not found: {image_path}")
    world_path = _resolved_path(
        capture.get("world_state_path"), base_dir=base_dir, field="world_state_path"
    )
    projection_path = _resolved_path(
        capture.get("projection_path"), base_dir=base_dir, field="projection_path"
    )
    world_state = json.loads(world_path.read_text(encoding="utf-8"))
    world_errors = validate_world_state(world_state)
    if world_errors:
        raise ValueError("invalid WorldState: " + "; ".join(world_errors))
    if world_state["frame_id"] != frame_id:
        raise ValueError("capture frame_id and WorldState frame_id must match")
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    projection_errors = validate_projection_record(
        projection,
        frame_id=frame_id,
        camera_name=camera_name,
    )
    if projection_errors:
        raise ValueError("invalid projection record: " + "; ".join(projection_errors))
    prompt = _render_prompt(prompt_template, frame_id=frame_id, camera_name=camera_name)
    return {
        "frame_id": frame_id,
        "source": "carla",
        "camera_name": camera_name,
        "image_path": str(image_path),
        "world_state_path": str(world_path),
        "projection_path": str(projection_path),
        "ground_truth_objects": projection["objects"],
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    }


def build_manifest(
    captures: Iterable[Mapping[str, Any]],
    *,
    base_dir: Path,
    prompt_template: str,
    require_images: bool = True,
) -> list[dict[str, Any]]:
    records = [
        build_manifest_record(
            capture,
            base_dir=base_dir,
            prompt_template=prompt_template,
            require_image=require_images,
        )
        for capture in captures
    ]
    frame_ids = [record["frame_id"] for record in records]
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("capture index contains duplicate frame_id values")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-index", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-missing-images", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    captures = read_jsonl(args.capture_index)
    prompt_template = args.prompt.read_text(encoding="utf-8")
    records = build_manifest(
        captures,
        base_dir=args.capture_index.parent,
        prompt_template=prompt_template,
        require_images=not args.allow_missing_images,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} CARLA frame records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
