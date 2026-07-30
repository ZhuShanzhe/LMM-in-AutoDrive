#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example usage of DeepFilterNet denoiser (official df module).

Usage:
    python denoiser_example.py
"""

import os
import logging
from optimization import DenoiseService, DenoiserConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def example_single():
    """Denoise a single file."""
    config = DenoiserConfig(
        model_name="DeepFilterNet3",
        output_sr=16000,
        output_dir="data"
    )
    service = DenoiseService(config)

    input_file = "data/example_noisy.wav"
    if not os.path.exists(input_file):
        logger.warning(f"Input file '{input_file}' not found. Please create one.")
        return

    result = service.denoise(
        audio=input_file,
        output_path="data/example_denoised.wav",
        output_json="result.json",
    )
    logger.info(f"Denoised file: {result['output_file']}")
    logger.info(f"Time: {result['processing_time_seconds']:.3f}s")


def example_batch():
    """Denoise multiple files."""
    config = DenoiserConfig(model_name="DeepFilterNet3", output_sr=16000)
    service = DenoiseService(config)

    files = ["noisy1.wav", "noisy2.wav", "noisy3.wav"]
    existing = [f for f in files if os.path.exists(f)]
    if not existing:
        logger.warning("No input files found. Skipping batch.")
        return

    outputs = service.denoise(
        audio=existing,
        output_json="batch_result.json",
    )
    for out in outputs:
        logger.info(f"Output: {out}")
    logger.info(f"Total batch time: {service.last_batch_time:.3f}s")


if __name__ == "__main__":
    example_single()
    # example_batch()
