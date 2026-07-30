from __future__ import annotations

import argparse
import json
from pathlib import Path

from structured_command_parser import ModernBertCommandService


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "models" / "modernbert-drive-command-compositional"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse one English driving command with ModernBERT"
    )
    parser.add_argument("text", help="English text produced by the translation module")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Fine-tuned model directory inside the submission package",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--request-id")
    parser.add_argument("--modality", choices=["TEXT", "VOICE"], default="TEXT")
    args = parser.parse_args()
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
