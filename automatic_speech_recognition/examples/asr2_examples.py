#!/usr/bin/env python
# test_qwen3_single.py
"""
Test Qwen3-ASR pipeline on a single audio file, with optional translation.
Displays ASR result and translation.

Usage:
    python test_qwen3_single.py --audio path/to/audio.wav [--translate]
"""

import argparse
import logging
import json

from src.pipeline2 import ASR2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-ASR pipeline on a single audio file")
    parser.add_argument("--audio", required=True, help="Path to input WAV file")
    parser.add_argument("--translate", action="store_true", help="Enable translation to English")
    parser.add_argument("--device", default="cuda:0", help="Device (cuda:0 / cpu)")
    parser.add_argument("--model_path", default="./models/Qwen3-ASR-1.7B", help="Path to local Qwen3-ASR model")
    parser.add_argument("--language", default="Chinese", help="ASR language (e.g., Chinese, English, Cantonese)")
    parser.add_argument("--output", help="Optional JSON output file")
    args = parser.parse_args()

    pipeline = ASR2(
        asr_load_type="local",
        asr_model_path=args.model_path,
        asr_device=args.device,
        asr_language=args.language,
        trans_load_type="local",
        trans_model_path="models/Qwen2.5-3B-Instruct",
        output_dir="data",
    )

    logger.info(f"Processing: {args.audio}")

    result = pipeline.process(
        audio_path=args.audio,
        translate=args.translate,
        output_json=args.output,
        language=args.language,
    )

    print("\n" + "="*60)
    print("Qwen3-ASR PIPELINE RESULT")
    print("="*60)
    print(f"Audio file: {result['audio_file']}")
    print(f"Chinese text: {result['chinese_text']}")
    if args.translate:
        print(f"English translation: {result['english_translation']}")
    print(f"ASR time: {result['asr_processing_time_seconds']:.3f}s")
    if args.translate:
        print(f"Translation time: {result['translation_time_seconds']:.3f}s")
    print(f"Total time: {result['total_time_seconds']:.3f}s")
    print("="*60)


if __name__ == "__main__":
    main()
