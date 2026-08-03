from __future__ import annotations

import argparse
import json
import os

from structured_command_parser import ModernBertCommandService


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse one English driving command with ModernBERT"
    )
    parser.add_argument("text", help="English text produced by the translation module")
    parser.add_argument(
        "--model",
        default=os.environ.get("MODERNBERT_MODEL_PATH"),
        help="Fine-tuned model directory; defaults to MODERNBERT_MODEL_PATH",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--request-id")
    parser.add_argument("--modality", choices=["TEXT", "VOICE"], default="TEXT")
    args = parser.parse_args()
    if not args.model:
        parser.error("--model or MODERNBERT_MODEL_PATH is required")

    service = ModernBertCommandService(args.model, device=args.device)
    service.warmup()
    result = service.parse_text(
        args.text,
        request_id=args.request_id,
        modality=args.modality,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
