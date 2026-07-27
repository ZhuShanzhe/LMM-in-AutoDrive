"""Run a focused Qwen pass for traffic-light and traffic-sign grounding."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from scene_understanding.core.normalize_scene_output import normalize_bbox_coordinates
from scene_understanding.core.prepare_nuscenes_samples import render_prompt
from scene_understanding.core.run_qwen_scene_inference import (
    append_jsonl,
    generate_one,
    load_backend,
    read_jsonl,
    select_frame_indices,
)
from scene_understanding.core.validate_scene_output import parse_json_text


LIGHT_STATES = {"red", "yellow", "green", "off", "unknown"}
SIGN_TYPES = {"no_entry", "stop", "yield", "speed_limit", "one_way", "other", "unknown"}


def parse_grounding_text(text: str) -> Any:
    """Parse strict JSON or extract the first JSON value from short model prose."""

    try:
        return parse_json_text(text)
    except json.JSONDecodeError as strict_error:
        decoder = json.JSONDecoder()
        starts = [index for index, character in enumerate(text) if character in "[{"]
        if not starts:
            raise strict_error
        try:
            value, _ = decoder.raw_decode(text[starts[0] :])
            return value
        except json.JSONDecodeError:
            # Never accept a complete inner object from a truncated outer response.
            raise strict_error


def _confidence(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if 0 < value <= 1 else None


def _sign_type(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
    if "no entry" in text or "do not enter" in text:
        return "no_entry"
    if "speed" in text and "limit" in text:
        return "speed_limit"
    if "one way" in text:
        return "one_way"
    if "stop" in text:
        return "stop"
    if "yield" in text or "give way" in text:
        return "yield"
    return "unknown" if not text else "other"


def _item_kind(item: dict[str, Any]) -> str | None:
    label = str(item.get("label") or item.get("category") or item.get("type") or "").lower()
    if "light" in label or "signal" in label:
        return "light"
    if "sign" in label or _sign_type(label) not in {"other", "unknown"}:
        return "sign"
    return None


def _light_state(item: dict[str, Any]) -> str:
    text = " ".join(
        str(item.get(key) or "").strip().lower() for key in ("state", "color", "label")
    )
    for state in ("red", "yellow", "green", "off"):
        if state in text:
            return state
    return "unknown"


def normalize_grounding_output(
    data: Any,
    *,
    frame_id: str,
    source: str,
    camera_name: str,
    processed_width: int,
    processed_height: int,
) -> tuple[dict[str, Any], list[str]]:
    """Canonicalize expected or native Qwen grounding JSON with an audit trail."""

    actions: list[str] = []
    light_items: list[Any] = []
    sign_items: list[Any] = []
    if isinstance(data, dict):
        if isinstance(data.get("traffic_lights"), list):
            light_items.extend(data["traffic_lights"])
        if isinstance(data.get("traffic_signs"), list):
            sign_items.extend(data["traffic_signs"])
        native_items = data.get("objects")
        if isinstance(native_items, list):
            for item in native_items:
                if isinstance(item, dict) and _item_kind(item) == "light":
                    light_items.append(item)
                elif isinstance(item, dict) and _item_kind(item) == "sign":
                    sign_items.append(item)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and _item_kind(item) == "light":
                light_items.append(item)
            elif isinstance(item, dict) and _item_kind(item) == "sign":
                sign_items.append(item)

    canonical: dict[str, Any] = {
        "schema_version": "1.0",
        "frame_id": frame_id,
        "source": source,
        "camera_name": camera_name,
        "traffic_lights": [],
        "traffic_signs": [],
    }

    for index, item in enumerate(light_items, start=1):
        if not isinstance(item, dict):
            continue
        bbox = normalize_bbox_coordinates(
            item.get("bbox_2d"),
            processed_width=processed_width,
            processed_height=processed_height,
        )
        state = _light_state(item)
        confidence = _confidence(item.get("confidence"))
        if confidence is None:
            actions.append(f"traffic_lights[{index - 1}]: confidence unavailable; stored null")
        if bbox is not None and bbox != item.get("bbox_2d"):
            actions.append(f"traffic_lights[{index - 1}]: normalized processed-image bbox")
        canonical["traffic_lights"].append(
            {
                "grounding_id": f"tc_light_{index:03d}",
                "bbox_2d": bbox if bbox is not None else item.get("bbox_2d"),
                "state": state,
                "confidence": confidence,
            }
        )

    for index, item in enumerate(sign_items, start=1):
        if not isinstance(item, dict):
            continue
        bbox = normalize_bbox_coordinates(
            item.get("bbox_2d"),
            processed_width=processed_width,
            processed_height=processed_height,
        )
        sign_type = str(item.get("sign_type") or "").lower()
        if sign_type not in SIGN_TYPES:
            sign_type = _sign_type(item.get("label") or item.get("type"))
        confidence = _confidence(item.get("confidence"))
        if confidence is None:
            actions.append(f"traffic_signs[{index - 1}]: confidence unavailable; stored null")
        if bbox is not None and bbox != item.get("bbox_2d"):
            actions.append(f"traffic_signs[{index - 1}]: normalized processed-image bbox")
        canonical["traffic_signs"].append(
            {
                "grounding_id": f"tc_sign_{index:03d}",
                "bbox_2d": bbox if bbox is not None else item.get("bbox_2d"),
                "sign_type": sign_type,
                "confidence": confidence,
            }
        )
    return canonical, actions


def _validate_bbox(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, list) or len(value) != 4:
        errors.append(f"{path}: expected four normalized coordinates")
        return
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
        errors.append(f"{path}: coordinates must be numbers")
        return
    if any(v < 0 or v > 1 for v in value):
        errors.append(f"{path}: coordinates must be between 0 and 1")
        return
    if value[0] >= value[2] or value[1] >= value[3]:
        errors.append(f"{path}: expected x_min < x_max and y_min < y_max")


def validate_grounding_output(data: Any, record: dict[str, Any]) -> list[str]:
    """Validate canonical focused-grounding output."""

    errors: list[str] = []
    required = {
        "schema_version",
        "frame_id",
        "source",
        "camera_name",
        "traffic_lights",
        "traffic_signs",
    }
    if not isinstance(data, dict):
        return ["root: expected an object"]
    if set(data) != required:
        errors.append("root: fields do not match focused grounding schema")
    for key in ("frame_id", "source", "camera_name"):
        if data.get(key) != record.get(key):
            errors.append(f"{key}: does not match manifest metadata")
    if data.get("schema_version") != "1.0":
        errors.append("schema_version: expected '1.0'")

    for name, prefix, value_key, allowed in (
        ("traffic_lights", "tc_light_", "state", LIGHT_STATES),
        ("traffic_signs", "tc_sign_", "sign_type", SIGN_TYPES),
    ):
        items = data.get(name)
        if not isinstance(items, list):
            errors.append(f"{name}: expected an array")
            continue
        for index, item in enumerate(items):
            path = f"{name}[{index}]"
            if not isinstance(item, dict):
                errors.append(f"{path}: expected an object")
                continue
            expected_keys = {"grounding_id", "bbox_2d", value_key, "confidence"}
            if set(item) != expected_keys:
                errors.append(f"{path}: fields do not match schema")
            grounding_id = item.get("grounding_id")
            if not isinstance(grounding_id, str) or not re.fullmatch(
                re.escape(prefix) + r"[0-9]{3,}", grounding_id
            ):
                errors.append(f"{path}.grounding_id: invalid ID")
            _validate_bbox(item.get("bbox_2d"), f"{path}.bbox_2d", errors)
            if item.get(value_key) not in allowed:
                errors.append(f"{path}.{value_key}: invalid value")
            confidence = item.get("confidence")
            if confidence is not None and (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0 < confidence <= 1
            ):
                errors.append(f"{path}.confidence: expected null or a number in (0, 1]")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "prompts" / "traffic_control_grounding.txt",
    )
    parser.add_argument("--frame-index", type=int, action="append")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--min-visual-tokens", type=int, default=256)
    parser.add_argument("--max-visual-tokens", type=int, default=1024)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    template = args.prompt_template.read_text(encoding="utf-8")
    records = select_frame_indices(read_jsonl(args.manifest), args.frame_index)
    for record in records:
        record["prompt"] = render_prompt(
            template,
            frame_id=record["frame_id"],
            source=record["source"],
            camera_name=record["camera_name"],
        )

    print(f"Loading model from {args.model_path}")
    model, processor, torch_module = load_backend(
        args.model_path,
        min_visual_tokens=args.min_visual_tokens,
        max_visual_tokens=args.max_visual_tokens,
    )
    valid_count = 0
    for index, record in enumerate(records, start=1):
        print(f"[{index}/{len(records)}] {record['frame_id']}")
        raw_output: str | None = None
        try:
            raw_output, elapsed, memory, processed_size = generate_one(
                record,
                model=model,
                processor=processor,
                torch_module=torch_module,
                max_new_tokens=args.max_new_tokens,
            )
            raw_parsed = parse_grounding_text(raw_output)
            parsed, actions = normalize_grounding_output(
                raw_parsed,
                frame_id=record["frame_id"],
                source=record["source"],
                camera_name=record["camera_name"],
                processed_width=processed_size[0],
                processed_height=processed_size[1],
            )
            errors = validate_grounding_output(parsed, record)
            status = "valid" if not errors else "invalid"
            valid_count += int(status == "valid")
            append_jsonl(
                args.output,
                {
                    "frame_id": record["frame_id"],
                    "source": record["source"],
                    "camera_name": record["camera_name"],
                    "image_path": record["image_path"],
                    "prompt_sha256": hashlib.sha256(record["prompt"].encode()).hexdigest(),
                    "status": status,
                    "elapsed_seconds": round(elapsed, 4),
                    "peak_memory_gib": round(memory, 4),
                    "processed_image_size": {
                        "width": processed_size[0],
                        "height": processed_size[1],
                    },
                    "raw_output": raw_output,
                    "raw_parsed_output": raw_parsed,
                    "parsed_output": parsed,
                    "normalization_actions": actions,
                    "validation_errors": errors,
                },
            )
            print(
                f"  status={status} lights={len(parsed['traffic_lights'])} "
                f"signs={len(parsed['traffic_signs'])} elapsed={elapsed:.2f}s errors={len(errors)}"
            )
        except Exception as exc:
            append_jsonl(
                args.output,
                {
                    "frame_id": record["frame_id"],
                    "source": record["source"],
                    "camera_name": record["camera_name"],
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "raw_output": raw_output,
                },
            )
            print(f"  ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.fail_fast:
                raise
    print(f"Finished {len(records)} frames; valid focused outputs: {valid_count}/{len(records)}")
    print(f"Results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
