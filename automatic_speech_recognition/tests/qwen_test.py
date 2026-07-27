import os
import logging
from typing import List, Dict, Any

from utils.data_loader import load_test_json, save_results_to_json
from utils.evaluator import ASREvaluator
from src.pipeline2 import Qwen3ASRPipeline
from src.config import Qwen3PipelineConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_integration_test(
    dataset_json: str,
    output_dir: str = "test_outputs",
    asr_device: str = "cuda:0",
    load_type: str = "local",            # "custom" or "local"
    model_path: str = "./models/Qwen3-ASR-1.7B",
    language: str = "Chinese",
    enable_enhancement: bool = False,
    enable_dialect_mapping: bool = False,
) -> Dict[str, Any]:
    """
    Run ASR pipeline on all samples in dataset and evaluate.

    Args:
        dataset_json: Path to JSON file containing test samples.
        output_dir: Directory to save per-sample results and final summary.
        asr_device: Device for ASR.
        load_type: 'custom' or 'local'.
        model_path: Local model path (if load_type=='local').
        language: Language hint (full name, e.g., "Chinese", "English").
        enable_enhancement: Enable noise reduction.
        enable_dialect_mapping: Enable dialect mapping.

    Returns:
        Evaluation results dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)

    samples = load_test_json(dataset_json)
    logger.info(f"Loaded {len(samples)} samples from {dataset_json}")

    config = Qwen3PipelineConfig(
        load_type=load_type,
        model_path=model_path,
        device=asr_device,
        language=language,
        enable_enhancement=enable_enhancement,
        enable_dialect_mapping=enable_dialect_mapping,
    )
    pipeline = Qwen3ASRPipeline(config=config)

    hypotheses = []
    refs = []
    per_sample_results = []
    total = len(samples)

    for idx, sample in enumerate(samples, 1):
        audio_path = sample['audio_file']
        reference = sample['reference']
        uid = sample['id']

        logger.info(f"[{idx}/{total}] Processing: {audio_path}")

        if not os.path.exists(audio_path):
            logger.warning(f"File not found: {audio_path}, skipping.")
            hypotheses.append("")
            refs.append(reference)
            per_sample_results.append({
                "id": uid,
                "reference": reference,
                "hypothesis": "",
                "error": "File not found"
            })
            continue

        try:
            result = pipeline.process(audio_path, language=language)
            hypothesis = result.get("text", "")
        except Exception as e:
            logger.error(f"Failed to process {audio_path}: {e}")
            hypothesis = ""

        hypotheses.append(hypothesis)
        refs.append(reference)
        per_sample_results.append({
            "id": uid,
            "reference": reference,
            "hypothesis": hypothesis,
            "original": sample.get("original", "")
        })

    output_file = os.path.join(output_dir, "ASR_results_qwen3.json")
    output = {
        "dataset": dataset_json,
        "total_samples": len(samples),
        "processed_samples": len([h for h in hypotheses if h != ""]),
        "per_sample": per_sample_results,
    }
    save_results_to_json(output, output_file)

    evaluator = ASREvaluator()
    eval_results = evaluator.evaluate_lists(refs, hypotheses)

    final_output = {
        "dataset": dataset_json,
        "total_samples": len(samples),
        "processed_samples": len([h for h in hypotheses if h != ""]),
        "overall_metrics": eval_results["overall"],
        "per_sample": per_sample_results,
        "raw_evaluation": eval_results
    }

    summary_file = os.path.join(output_dir, "test_summary_qwen3.json")
    save_results_to_json(final_output, summary_file)

    evaluator.print_summary()

    return final_output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3 ASR Integration Test")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON file")
    parser.add_argument("--output_dir", default="test_outputs", help="Output directory")
    parser.add_argument("--asr_device", default="cuda:0", help="Device for ASR")
    parser.add_argument("--asr_mode", default="single", choices=["single", "speaker"])
    parser.add_argument("--load_type", default="custom", choices=["custom", "local"])
    parser.add_argument("--model_path", default=None)
    args = parser.parse_args()

    run_integration_test(
        dataset_json=args.dataset,
        output_dir=args.output_dir,
        asr_device=args.asr_device,
        load_type=args.load_type,
        model_path=args.model_path,
        language="Chinese",
        enable_enhancement=False,
        enable_dialect_mapping=False,
    )
