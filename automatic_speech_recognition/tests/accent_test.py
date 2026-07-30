import os
import argparse
import logging
import time
from typing import Dict, List, Tuple, Optional, Any

from utils.data_loader import save_results_to_json
from utils.evaluator import ASREvaluator
from src import ASR as FunASR_Pipeline
from src import ASRConfig as FunASRConfig
from src import ASR2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DIALECT_MAP = {
    "Dongbei": "Dongbei Dialect Speech Corpus for TTS",
    "Guangzhou": "Guangzhou Cantonese Speech Corpus for TTS",
    "Henan": "Henan Dialect Speech Corpus for TTS",
    "Jiangsu": "Jiangsu Dialect Speech Corpus for TTS",
    "Sichuan": "Sichuan Dialect Speech Corpus for TTS",
}


def load_dialect_dataset(dialect_dir: str, txt_file: str = "ProsodyLabeling/txt.txt") -> List[Tuple[str, str]]:
    """
    Load dialect dataset from a directory.

    Args:
        dialect_dir: Path to the dialect root directory.
        txt_file: Relative path to the transcript file (default: ProsodyLabeling/txt.txt).

    Returns:
        List of (wav_path, transcript) tuples.
    """
    txt_path = os.path.join(dialect_dir, txt_file)
    if not os.path.exists(txt_path):
        logger.warning(f"Transcript file not found: {txt_path}")
        return []

    samples = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) != 2:
                logger.warning(f"Skipping malformed line: {line}")
                continue
            wav_filename, text = parts
            wav_path = os.path.join(dialect_dir, wav_filename)

            if not os.path.exists(wav_path):
                alt_path = os.path.join(dialect_dir, "wav", wav_filename)
                if os.path.exists(alt_path):
                    wav_path = alt_path
                else:
                    logger.debug(f"Wav file not found: {wav_path}")
                    continue
            samples.append((wav_path, text))
    return samples


def run_asr_evaluation(
    samples: List[Tuple[str, str]],
    asr_pipeline,
    translate: bool = False,
) -> Tuple[List[str], List[str], float]:
    """
    Run ASR on a list of samples.

    Args:
        samples: List of (wav_path, reference_text).
        asr_pipeline: The ASR pipeline instance (ASR or ASR2).
        translate: Unused (kept for compatibility).

    Returns:
        (references, hypotheses, total_time)
    """
    references = []
    hypotheses = []
    total_time = 0.0

    for wav_path, ref_text in samples:
        if not os.path.exists(wav_path):
            logger.warning(f"File not found: {wav_path}, skipping.")
            references.append(ref_text)
            hypotheses.append("")
            continue

        start = time.perf_counter()
        try:
            result = asr_pipeline.process(wav_path, translate=False)
            hyp_text = result.get("chinese_text", "")
        except Exception as e:
            logger.error(f"ASR failed on {wav_path}: {e}")
            hyp_text = ""
        elapsed = time.perf_counter() - start
        total_time += elapsed

        references.append(ref_text)
        hypotheses.append(hyp_text)

    return references, hypotheses, total_time


def evaluate_dialect(
    dialect_name: str,
    samples: List[Tuple[str, str]],
    asr_pipeline,
    asr2_pipeline,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate a dialect using both ASR engines.

    Args:
        dialect_name: Name of the dialect.
        samples: List of (wav_path, reference_text).
        asr_pipeline: FunASR pipeline instance.
        asr2_pipeline: Qwen3-ASR pipeline instance.
        max_samples: Limit number of samples per dialect.

    Returns:
        Dict with evaluation results.
    """
    if max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]
        logger.info(f"Limiting {dialect_name} to {max_samples} samples.")

    logger.info(f"Evaluating {dialect_name} with {len(samples)} samples.")

    refs, hyps_asr, time_asr = run_asr_evaluation(samples, asr_pipeline)
    refs2, hyps_asr2, time_asr2 = run_asr_evaluation(samples, asr2_pipeline)

    evaluator = ASREvaluator()
    results_asr = evaluator.evaluate_lists(refs, hyps_asr)
    results_asr2 = evaluator.evaluate_lists(refs2, hyps_asr2)

    return {
        "dialect": dialect_name,
        "num_samples": len(samples),
        "asr": {
            "cer": results_asr["overall"]["average_cer"],
            "wer": results_asr["overall"]["average_wer"],
            "time_total": time_asr,
        },
        "asr2": {
            "cer": results_asr2["overall"]["average_cer"],
            "wer": results_asr2["overall"]["average_wer"],
            "time_total": time_asr2,
        },
        "per_sample_asr": [
            {"reference": r, "hypothesis": h} for r, h in zip(refs, hyps_asr)
        ],
        "per_sample_asr2": [
            {"reference": r, "hypothesis": h} for r, h in zip(refs2, hyps_asr2)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ASR on dialects.")
    parser.add_argument("--data_root", required=True,
                        help="Root directory containing dialect subdirectories.")
    parser.add_argument("--output_dir", default="dialect_results",
                        help="Directory to save evaluation results.")
    parser.add_argument("--asr_device", default="cuda:0",
                        help="Device for ASR (cuda:0 / cpu).")
    parser.add_argument("--model_path", default="./models/Qwen3-ASR-1.7B",
                        help="Path to Qwen3-ASR model.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples per dialect (optional).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Initializing FunASR pipeline...")
    funasr_config = FunASRConfig(
        asr_mode="single",
        asr_device=args.asr_device,
    )
    asr_pipeline = FunASR_Pipeline(config=funasr_config)

    logger.info("Initializing Qwen3-ASR pipeline...")
    asr2_pipeline = ASR2(
        asr_load_type="local",
        asr_model_path=args.model_path,
        asr_device=args.asr_device,
        asr_language="Chinese",
        trans_load_type="custom",   # translation disabled
        trans_model_name="Qwen/Qwen2.5-3B-Instruct",
        output_dir=args.output_dir,
    )

    all_results = []
    for dialect_name, subdir in DIALECT_MAP.items():
        dialect_path = os.path.join(args.data_root, subdir)
        if not os.path.isdir(dialect_path):
            logger.warning(f"Dialect directory not found: {dialect_path}")
            continue

        samples = load_dialect_dataset(dialect_path)
        if not samples:
            logger.warning(f"No samples found for {dialect_name}.")
            continue

        result = evaluate_dialect(
            dialect_name=dialect_name,
            samples=samples,
            asr_pipeline=asr_pipeline,
            asr2_pipeline=asr2_pipeline,
            max_samples=args.max_samples,
        )
        all_results.append(result)

    summary = {
        "total_dialects": len(all_results),
        "results": all_results,
    }

    print("\n" + "=" * 80)
    print("Dialect ASR Evaluation Summary")
    print("=" * 80)
    print(f"{'Dialect':<12} {'#Samples':<10} {'ASR CER':<10} {'ASR2 CER':<10} {'ASR WER':<10} {'ASR2 WER':<10}")
    print("-" * 80)
    for r in all_results:
        d = r["dialect"]
        n = r["num_samples"]
        cer_asr = r["asr"]["cer"]
        cer_asr2 = r["asr2"]["cer"]
        wer_asr = r["asr"]["wer"]
        wer_asr2 = r["asr2"]["wer"]
        print(f"{d:<12} {n:<10} {cer_asr:<10.4f} {cer_asr2:<10.4f} {wer_asr:<10.4f} {wer_asr2:<10.4f}")
    print("=" * 80)

    summary_file = os.path.join(args.output_dir, "dialect_eval_summary.json")
    save_results_to_json(summary, summary_file)
    logger.info(f"Summary saved to {summary_file}")

    for r in all_results:
        dialect = r["dialect"]
        detail_file = os.path.join(args.output_dir, f"{dialect}_details.json")
        save_results_to_json(r, detail_file)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
