from __future__ import annotations

import argparse
import json

from structured_command_parser import HybridCommandParser


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a Chinese driving command")
    parser.add_argument("text", help="Chinese driving command")
    parser.add_argument("--modality", choices=["TEXT", "VOICE"], default="TEXT")
    parser.add_argument("--model", help="Local Qwen model directory")
    args = parser.parse_args()

    command_parser = HybridCommandParser(model_path=args.model)
    result = command_parser.parse(args.text, modality=args.modality)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

