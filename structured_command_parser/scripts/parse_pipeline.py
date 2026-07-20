from __future__ import annotations

import argparse
import json

from structured_command_parser import ChineseEnglishCommandPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate a Chinese driving command and parse the English command"
    )
    parser.add_argument("text", help="Chinese ASR text")
    parser.add_argument("--model", help="Shared model path for both stages")
    parser.add_argument("--translator-model")
    parser.add_argument("--parser-model")
    parser.add_argument("--modality", choices=["TEXT", "VOICE"], default="VOICE")
    args = parser.parse_args()

    translator_model = args.translator_model or args.model
    parser_model = args.parser_model or args.model
    if not translator_model or not parser_model:
        parser.error("use --model, or provide both --translator-model and --parser-model")
    pipeline = ChineseEnglishCommandPipeline(translator_model, parser_model)
    result = pipeline.parse(args.text, modality=args.modality)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
