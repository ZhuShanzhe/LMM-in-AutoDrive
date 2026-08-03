#!/usr/bin/env python
# test_funasr_single.py
"""
Test FunASR pipeline on a single audio file, with optional translation.
Displays ASR result and translation.

Usage:
    python test_funasr_single.py --audio path/to/audio.wav [--translate]
"""

import argparse
import logging
import json
from pathlib import Path

from src import ASR
from src import ASRConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test FunASR pipeline on a single audio file")
    parser.add_argument("--audio", required=True, help="Path to input WAV file")
    parser.add_argument("--translate", action="store_true", help="Enable translation to English")
    parser.add_argument("--device", default="cuda:0", help="Device (cuda:0 / cpu)")
    parser.add_argument("--output", help="Optional JSON output file")
    args = parser.parse_args()

    config = ASRConfig(
        asr_mode="single",
        asr_device=args.device,
        trans_load_type="local",
        trans_model_path="models/Qwen2.5-3B-Instruct",
        output_dir="data",
    )
    pipeline = ASR(config=config)

    logger.info(f"Processing: {args.audio}")
    result = pipeline.process(
        audio_path=args.audio,
        translate=args.translate,
        output_json=args.output
    )

    print("\n" + "="*60)
    print("FUNASR PIPELINE RESULT")
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


