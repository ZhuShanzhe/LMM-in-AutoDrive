"""Build a CAM_FRONT inference manifest from DriveLM nuScenes annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CAMERA = "CAM_FRONT"
DEFAULT_IMAGE_WIDTH = 1600
DEFAULT_IMAGE_HEIGHT = 900


def render_prompt(template: str, *, frame_id: str, source: str, camera_name: str) -> str:
    """Replace only our three metadata placeholders, leaving JSON braces untouched."""

    return (
        template.replace("{frame_id}", frame_id)
        .replace("{source}", source)
        .replace("{camera_name}", camera_name)
    )


def resolve_drivelm_image_path(image_root: Path, annotation_path: str) -> Path:
    """Resolve DriveLM paths such as ../nuscenes/samples/CAM_FRONT/image.jpg."""

    normalized = annotation_path.replace("\\", "/")
    if normalized.startswith("../"):
        normalized = "data/" + normalized[3:]
    return image_root / normalized


def normalize_category(raw_category: Any, visual_description: Any) -> str:
    """Map DriveLM labels conservatively into the scene-output category vocabulary."""

    category = str(raw_category or "").strip().lower()
    description = str(visual_description or "").strip().lower()

    if "pedestrian" in category or "person" in category:
        return "pedestrian"
    if "motorcycle" in category or "motorbike" in category:
        return "motorcycle"
    if "cyclist" in category or "bicycle" in category or "bike" in category:
        return "cyclist"
    if "vehicle" in category or any(
        name in category for name in ("car", "truck", "bus", "trailer", "van")
    ):
        return "vehicle"
    if "barrier" in category or "barrier" in description:
        return "road_barrier"
    if "cone" in category or "cone" in description:
        return "traffic_cone"
    if "traffic" in category:
        if "light" in description or "signal" in description:
            return "traffic_light"
        if any(
            term in description
            for term in (
                "sign",
                "no entry",
                "do not enter",
                "stop",
                "speed limit",
                "give way",
                "yield",
                "one way",
            )
        ):
            return "traffic_sign"
        return "other"
    if "animal" in category:
        return "animal"
    return "other" if category or description else "unknown"


def normalize_bbox(
    bbox: Any,
    *,
    image_width: int,
    image_height: int,
    object_tag: str,
) -> list[float]:
    """Convert a pixel bbox to normalized coordinates and reject malformed truth data."""

    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"{object_tag}: expected a four-value 2d_bbox")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox):
        raise ValueError(f"{object_tag}: 2d_bbox values must be numbers")

    x_min, y_min, x_max, y_max = (float(value) for value in bbox)
    if not (0 <= x_min < x_max <= image_width):
        raise ValueError(f"{object_tag}: x coordinates fall outside image width {image_width}")
    if not (0 <= y_min < y_max <= image_height):
        raise ValueError(f"{object_tag}: y coordinates fall outside image height {image_height}")

    return [
        round(x_min / image_width, 6),
        round(y_min / image_height, 6),
        round(x_max / image_width, 6),
        round(y_max / image_height, 6),
    ]


def camera_from_object_tag(object_tag: str) -> str | None:
    """Return the camera token from tags like <c3,CAM_FRONT,1043.2,82.2>."""

    stripped = object_tag.strip("<>")
    parts = stripped.split(",")
    if len(parts) < 2:
        return None
    return parts[1]


def iter_frames(annotations: Any) -> Iterable[tuple[str, str, dict[str, Any], str]]:
    """Yield scene ID, frame token, frame data, and scene description."""

    if not isinstance(annotations, dict):
        raise ValueError("DriveLM annotations must be a JSON object keyed by scene ID")

    for scene_id, scene_data in annotations.items():
        if not isinstance(scene_data, dict):
            raise ValueError(f"scene {scene_id}: expected an object")
        key_frames = scene_data.get("key_frames")
        if not isinstance(key_frames, dict):
            raise ValueError(f"scene {scene_id}: missing key_frames object")
        scene_description = str(scene_data.get("scene_description") or "")

        for frame_token, frame_data in key_frames.items():
            if not isinstance(frame_data, dict):
                raise ValueError(f"frame {frame_token}: expected an object")
            yield str(scene_id), str(frame_token), frame_data, scene_description


def build_manifest(
    annotations: Any,
    *,
    image_root: Path,
    prompt_template: str,
    camera_name: str = DEFAULT_CAMERA,
    limit: int | None = 30,
    image_width: int = DEFAULT_IMAGE_WIDTH,
    image_height: int = DEFAULT_IMAGE_HEIGHT,
    require_images: bool = False,
) -> list[dict[str, Any]]:
    """Create one deterministic inference record per DriveLM keyframe."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive or omitted")

    records: list[dict[str, Any]] = []
    for scene_id, frame_token, frame_data, scene_description in iter_frames(annotations):
        image_paths = frame_data.get("image_paths")
        if not isinstance(image_paths, dict) or camera_name not in image_paths:
            continue

        frame_id = f"{scene_id}_{frame_token}"
        image_path = resolve_drivelm_image_path(image_root, str(image_paths[camera_name]))
        if require_images and not image_path.is_file():
            raise FileNotFoundError(f"{frame_id}: image not found: {image_path}")

        key_object_infos = frame_data.get("key_object_infos") or {}
        if not isinstance(key_object_infos, dict):
            raise ValueError(f"{frame_id}: key_object_infos must be an object")

        ground_truth_objects: list[dict[str, Any]] = []
        for object_tag, object_info in key_object_infos.items():
            if camera_from_object_tag(str(object_tag)) != camera_name:
                continue
            if not isinstance(object_info, dict):
                raise ValueError(f"{frame_id}/{object_tag}: expected an object")

            bbox_pixels = object_info.get("2d_bbox")
            ground_truth_objects.append(
                {
                    "object_tag": str(object_tag),
                    "category_raw": object_info.get("Category"),
                    "category": normalize_category(
                        object_info.get("Category"),
                        object_info.get("Visual_description"),
                    ),
                    "status_raw": object_info.get("Status"),
                    "visual_description": object_info.get("Visual_description"),
                    "bbox_2d_pixels": bbox_pixels,
                    "bbox_2d": normalize_bbox(
                        bbox_pixels,
                        image_width=image_width,
                        image_height=image_height,
                        object_tag=str(object_tag),
                    ),
                }
            )

        qa_data = frame_data.get("QA") or {}
        qa_counts = {
            name: len(qa_data.get(name, [])) if isinstance(qa_data, dict) else 0
            for name in ("perception", "prediction", "planning", "behavior")
        }

        records.append(
            {
                "manifest_version": "1.0",
                "frame_id": frame_id,
                "source": "nuscenes",
                "camera_name": camera_name,
                "image_path": str(image_path),
                "scene_description": scene_description,
                "ground_truth_objects": ground_truth_objects,
                "qa_counts": qa_counts,
                "prompt_sha256": hashlib.sha256(prompt_template.encode("utf-8")).hexdigest(),
                "prompt": render_prompt(
                    prompt_template,
                    frame_id=frame_id,
                    source="nuscenes",
                    camera_name=camera_name,
                ),
            }
        )

        if limit is not None and len(records) >= limit:
            break

    return records


def write_jsonl(records: Iterable[dict[str, Any]], output_path: Path) -> int:
    """Write records as UTF-8 JSON Lines and return the number written."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path, help="DriveLM train_sample.json or compatible file")
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="DriveLM llama_adapter_v2_multimodal7b directory containing data/nuscenes",
    )
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "prompts" / "scene_understanding.txt",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL manifest")
    parser.add_argument("--camera-name", default=DEFAULT_CAMERA)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument(
        "--require-images",
        action="store_true",
        help="Fail when an image path does not exist",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    prompt_template = args.prompt_template.read_text(encoding="utf-8")
    records = build_manifest(
        annotations,
        image_root=args.image_root,
        prompt_template=prompt_template,
        camera_name=args.camera_name,
        limit=args.limit,
        image_width=args.image_width,
        image_height=args.image_height,
        require_images=args.require_images,
    )
    if not records:
        raise SystemExit(f"No frames found for camera {args.camera_name}")

    count = write_jsonl(records, args.output)
    object_count = sum(len(record["ground_truth_objects"]) for record in records)
    print(f"Wrote {count} frames and {object_count} front-camera truth objects to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
