"""Run MiniCPM-V scene understanding over the same manifest used by Qwen."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

from scene_understanding.core.normalize_scene_output import normalize_scene_output
from scene_understanding.core.run_qwen_scene_inference import (
    append_jsonl,
    completed_frame_ids,
    evaluate_output,
    read_jsonl,
    select_frame_indices,
    select_records,
)
from scene_understanding.core.validate_scene_output import validate_scene_output


def load_backend(model_path: Path):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        str(model_path),
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
    ).eval()
    return model, processor, torch


def generate_one(
    record: dict[str, Any],
    *,
    model: Any,
    processor: Any,
    torch_module: Any,
    max_new_tokens: int,
    downsample_mode: str,
    max_slice_nums: int,
) -> tuple[str, float, float, tuple[int, int]]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": record["image_path"]},
                {"type": "text", "text": record["prompt"]},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        downsample_mode=downsample_mode,
        max_slice_nums=max_slice_nums,
    ).to(model.device, dtype=model.dtype)
    with Image.open(record["image_path"]) as image:
        processed_size = image.size

    torch_module.cuda.reset_peak_memory_stats()
    torch_module.cuda.synchronize()
    started = time.perf_counter()
    with torch_module.inference_mode():
        generated_ids = model.generate(
            **inputs,
            downsample_mode=downsample_mode,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    torch_module.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started
    trimmed = [
        output[len(input_ids) :]
        for input_ids, output in zip(inputs.input_ids, generated_ids)
    ]
    text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    peak_memory_gib = torch_module.cuda.max_memory_allocated() / (1024**3)
    return text, elapsed_seconds, peak_memory_gib, processed_size


def infer_one(
    record: dict[str, Any],
    *,
    model: Any,
    processor: Any,
    torch_module: Any,
    model_path: Path,
    max_new_tokens: int,
    downsample_mode: str,
    max_slice_nums: int,
) -> dict[str, Any]:
    raw_output, elapsed_seconds, peak_memory_gib, processed_size = generate_one(
        record,
        model=model,
        processor=processor,
        torch_module=torch_module,
        max_new_tokens=max_new_tokens,
        downsample_mode=downsample_mode,
        max_slice_nums=max_slice_nums,
    )
    raw_parsed, raw_errors = evaluate_output(record, raw_output)
    if raw_parsed is None:
        parsed_output = None
        actions: list[str] = []
        errors = raw_errors
    else:
        parsed_output, actions = normalize_scene_output(
            raw_parsed,
            processed_width=processed_size[0],
            processed_height=processed_size[1],
        )
        errors = validate_scene_output(
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
            "prompt_sha256", hashlib.sha256(record["prompt"].encode("utf-8")).hexdigest()
        ),
        "status": "valid" if not errors else "invalid",
        "elapsed_seconds": round(elapsed_seconds, 4),
        "peak_memory_gib": round(peak_memory_gib, 4),
        "processed_image_size": {"width": processed_size[0], "height": processed_size[1]},
        "inference_config": {
            "backend": "minicpm_v",
            "model_path": str(model_path),
            "max_new_tokens": max_new_tokens,
            "downsample_mode": downsample_mode,
            "max_slice_nums": max_slice_nums,
            "do_sample": False,
        },
        "raw_output": raw_output,
        "raw_parsed_output": raw_parsed,
        "raw_validation_errors": raw_errors,
        "parsed_output": parsed_output,
        "normalization_actions": actions,
        "validation_errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--frame-index", type=int, action="append")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--downsample-mode", choices=("4x", "16x"), default="16x")
    parser.add_argument("--max-slice-nums", type=int, default=9)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model_path.is_dir():
        raise SystemExit(f"model directory not found: {args.model_path}")
    if args.output.exists() and not args.resume:
        raise SystemExit(f"output already exists; use --resume: {args.output}")
    records = select_frame_indices(read_jsonl(args.manifest), args.frame_index)
    completed = completed_frame_ids(args.output) if args.resume else set()
    selected = select_records(records, completed=completed, limit=args.limit)
    if not selected:
        print("No new frames to process")
        return 0

    print(f"Loading MiniCPM-V from {args.model_path}")
    model, processor, torch_module = load_backend(args.model_path)
    valid_count = 0
    for index, record in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {record['frame_id']}")
        try:
            result = infer_one(
                record,
                model=model,
                processor=processor,
                torch_module=torch_module,
                model_path=args.model_path,
                max_new_tokens=args.max_new_tokens,
                downsample_mode=args.downsample_mode,
                max_slice_nums=args.max_slice_nums,
            )
            append_jsonl(args.output, result)
            valid_count += int(result["status"] == "valid")
            print(
                f"  status={result['status']} elapsed={result['elapsed_seconds']:.2f}s "
                f"peak={result['peak_memory_gib']:.2f}GiB errors={len(result['validation_errors'])}"
            )
        except Exception as exc:
            append_jsonl(
                args.output,
                {
                    "frame_id": record["frame_id"],
                    "source": record["source"],
                    "camera_name": record["camera_name"],
                    "image_path": record["image_path"],
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            print(f"  ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.fail_fast:
                raise
    print(f"Finished {len(selected)} frames; schema-valid: {valid_count}/{len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
