"""Run deterministic Qwen2.5-VL scene understanding over a JSONL manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from scene_understanding.core.normalize_scene_output import normalize_scene_output
from scene_understanding.core.validate_scene_output import parse_json_text, validate_scene_output


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and reject malformed or duplicate frame records."""

    records: list[dict[str, Any]] = []
    frame_ids: set[str] = set()
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

            frame_id = record.get("frame_id")
            if not isinstance(frame_id, str) or not frame_id:
                raise ValueError(f"{path}:{line_number}: missing frame_id")
            if frame_id in frame_ids:
                raise ValueError(f"{path}:{line_number}: duplicate frame_id {frame_id}")

            image_path = record.get("image_path")
            if not isinstance(image_path, str) or not image_path:
                raise ValueError(f"{path}:{line_number}: missing image_path")
            if not Path(image_path).is_file():
                raise FileNotFoundError(f"{path}:{line_number}: image not found: {image_path}")

            prompt = record.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"{path}:{line_number}: missing prompt")

            frame_ids.add(frame_id)
            records.append(record)
    return records


def completed_frame_ids(path: Path) -> set[str]:
    """Return frame IDs already written to an existing inference output."""

    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid existing output: {exc}") from exc
            frame_id = record.get("frame_id")
            if isinstance(frame_id, str) and frame_id:
                completed.add(frame_id)
    return completed


def select_records(
    records: Iterable[dict[str, Any]],
    *,
    completed: set[str],
    limit: int | None,
) -> list[dict[str, Any]]:
    """Select unprocessed records in stable manifest order."""

    selected = [record for record in records if record["frame_id"] not in completed]
    return selected if limit is None else selected[:limit]


def select_frame_indices(
    records: list[dict[str, Any]], indices: list[int] | None
) -> list[dict[str, Any]]:
    """Select stable one-based manifest positions for targeted experiments."""

    if not indices:
        return records
    invalid = [index for index in indices if index < 1 or index > len(records)]
    if invalid:
        raise ValueError(f"frame indices outside 1..{len(records)}: {invalid}")
    if len(indices) != len(set(indices)):
        raise ValueError("frame indices must be unique")
    return [records[index - 1] for index in indices]


def build_conversation(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the standard Transformers multimodal chat structure."""

    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": record["image_path"]},
                {"type": "text", "text": record["prompt"]},
            ],
        }
    ]


def load_backend(
    model_path: Path,
    *,
    min_visual_tokens: int,
    max_visual_tokens: int,
):
    """Load the local Qwen model and processor without any network access."""

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this script inside a Slurm GPU allocation.")

    min_pixels = min_visual_tokens * 28 * 28
    max_pixels = max_visual_tokens * 28 * 28
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        use_fast=True,
        local_files_only=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    model.generation_config.temperature = None
    return model, processor, torch


def generate_one(
    record: dict[str, Any],
    *,
    model,
    processor,
    torch_module,
    max_new_tokens: int,
) -> tuple[str, float, float, tuple[int, int]]:
    """Generate one answer and return text, timings, memory, and processed image size."""

    conversation = build_conversation(record)
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    image_grid = inputs.image_grid_thw[0]
    patch_size = int(processor.image_processor.patch_size)
    processed_height = int(image_grid[1].item()) * patch_size
    processed_width = int(image_grid[2].item()) * patch_size

    torch_module.cuda.reset_peak_memory_stats()
    torch_module.cuda.synchronize()
    started = time.perf_counter()
    with torch_module.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    torch_module.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started

    generated_ids = [
        output[len(input_ids) :]
        for input_ids, output in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    peak_memory_gib = torch_module.cuda.max_memory_allocated() / (1024**3)
    return output_text, elapsed_seconds, peak_memory_gib, (processed_width, processed_height)


def evaluate_output(record: dict[str, Any], raw_output: str) -> tuple[Any | None, list[str]]:
    """Parse and validate generated JSON against manifest metadata."""

    try:
        parsed = parse_json_text(raw_output)
    except json.JSONDecodeError as exc:
        return None, [f"invalid JSON: {exc}"]

    errors = validate_scene_output(
        parsed,
        expected_frame_id=record["frame_id"],
        expected_source=record["source"],
        expected_camera_name=record["camera_name"],
    )
    return parsed, errors


def infer_one_record(
    record: dict[str, Any],
    *,
    model,
    processor,
    torch_module,
    model_path: Path,
    max_new_tokens: int,
    min_visual_tokens: int,
    max_visual_tokens: int,
) -> dict[str, Any]:
    """Run, normalize, and audit one manifest record with a loaded backend."""

    raw_output, elapsed_seconds, peak_memory_gib, processed_size = generate_one(
        record,
        model=model,
        processor=processor,
        torch_module=torch_module,
        max_new_tokens=max_new_tokens,
    )
    raw_parsed_output, raw_validation_errors = evaluate_output(record, raw_output)
    if raw_parsed_output is None:
        parsed_output = None
        normalization_actions: list[str] = []
        validation_errors = raw_validation_errors
    else:
        parsed_output, normalization_actions = normalize_scene_output(
            raw_parsed_output,
            processed_width=processed_size[0],
            processed_height=processed_size[1],
        )
        validation_errors = validate_scene_output(
            parsed_output,
            expected_frame_id=record["frame_id"],
            expected_source=record["source"],
            expected_camera_name=record["camera_name"],
        )
    return {
        "frame_id": record["frame_id"],
        "source": record["source"],
        "camera_name": record["camera_name"],
        "image_path": record["image_path"],
        "prompt_sha256": record.get(
            "prompt_sha256",
            hashlib.sha256(record["prompt"].encode("utf-8")).hexdigest(),
        ),
        "status": "valid" if not validation_errors else "invalid",
        "elapsed_seconds": round(elapsed_seconds, 4),
        "peak_memory_gib": round(peak_memory_gib, 4),
        "processed_image_size": {
            "width": processed_size[0],
            "height": processed_size[1],
        },
        "inference_config": {
            "model_path": str(model_path),
            "max_new_tokens": max_new_tokens,
            "min_visual_tokens": min_visual_tokens,
            "max_visual_tokens": max_visual_tokens,
            "do_sample": False,
        },
        "raw_output": raw_output,
        "raw_parsed_output": raw_parsed_output,
        "raw_validation_errors": raw_validation_errors,
        "parsed_output": parsed_output,
        "normalization_actions": normalization_actions,
        "validation_errors": validation_errors,
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append and flush one result so an interrupted run can resume safely."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Maximum new frames to process")
    parser.add_argument(
        "--frame-index",
        type=int,
        action="append",
        help="One-based manifest frame to process; may be repeated",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--min-visual-tokens", type=int, default=256)
    parser.add_argument("--max-visual-tokens", type=int, default=1024)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip frame IDs already present in the output file",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately after a per-frame inference error",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    if not (0 < args.min_visual_tokens <= args.max_visual_tokens):
        raise SystemExit("visual token bounds are invalid")
    if not args.model_path.is_dir():
        raise SystemExit(f"model directory not found: {args.model_path}")
    if args.output.exists() and not args.resume:
        raise SystemExit(f"output already exists; use --resume or choose another path: {args.output}")

    manifest = select_frame_indices(read_jsonl(args.manifest), args.frame_index)
    completed = completed_frame_ids(args.output) if args.resume else set()
    selected = select_records(manifest, completed=completed, limit=args.limit)
    if not selected:
        print("No new frames to process")
        return 0

    print(f"Loading model from {args.model_path}")
    model, processor, torch_module = load_backend(
        args.model_path,
        min_visual_tokens=args.min_visual_tokens,
        max_visual_tokens=args.max_visual_tokens,
    )

    valid_count = 0
    for index, record in enumerate(selected, start=1):
        frame_id = record["frame_id"]
        print(f"[{index}/{len(selected)}] {frame_id}")
        try:
            result = infer_one_record(
                record,
                model=model,
                processor=processor,
                torch_module=torch_module,
                model_path=args.model_path,
                max_new_tokens=args.max_new_tokens,
                min_visual_tokens=args.min_visual_tokens,
                max_visual_tokens=args.max_visual_tokens,
            )
            valid_count += int(result["status"] == "valid")
            append_jsonl(args.output, result)
            print(
                f"  status={result['status']} elapsed={result['elapsed_seconds']:.2f}s "
                f"peak_memory={result['peak_memory_gib']:.2f}GiB "
                f"errors={len(result['validation_errors'])}"
            )
        except Exception as exc:  # Keep an auditable record of failed frames.
            append_jsonl(
                args.output,
                {
                    "frame_id": frame_id,
                    "source": record["source"],
                    "camera_name": record["camera_name"],
                    "image_path": record["image_path"],
                    "prompt_sha256": record.get(
                        "prompt_sha256",
                        hashlib.sha256(record["prompt"].encode("utf-8")).hexdigest(),
                    ),
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            print(f"  ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.fail_fast:
                raise

    print(f"Finished {len(selected)} frames; schema-valid outputs: {valid_count}/{len(selected)}")
    print(f"Results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
