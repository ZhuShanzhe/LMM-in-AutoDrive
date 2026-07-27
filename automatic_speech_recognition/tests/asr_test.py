import os
import logging
from typing import List, Dict, Any

from utils.data_loader import load_test_json, save_results_to_json
from utils.evaluator import ASREvaluator
from src import ASR
from src import ASRConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_integration_test(
    dataset_json: str,
    output_dir: str = "test_outputs",
    asr_mode: str = "single",
    asr_device: str = "cuda:0",
) -> Dict[str, Any]:
    """
    Run ASR pipeline on all samples in dataset and evaluate.

    Args:
        dataset_json: Path to JSON file containing test samples.
        output_dir: Directory to save per-sample results and final summary.
        asr_mode: ASR mode ('single' or 'speaker').
        asr_device: Device for ASR.

    Returns:
        Evaluation results dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load dataset
    samples = load_test_json(dataset_json)
    logger.info(f"Loaded {len(samples)} samples from {dataset_json}")

    # 2. Initialize ASR pipeline
    config = ASRConfig(
        asr_mode=asr_mode,
        asr_device=asr_device,
    )
    pipeline = ASR(config=config)

    # 3. Process each sample (ASR only, translation disabled)
    hypotheses = []
    refs = []
    per_sample_results = []

    for idx, sample in enumerate(samples, 1):
        audio_path = sample['audio_file']
        reference = sample['reference']
        uid = sample['id']

        logger.info(f"Processing {idx}/{len(samples)}: {audio_path}")

        if not os.path.exists(audio_path):
            logger.warning(f"Audio file not found: {audio_path}, skipping.")
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
            result = pipeline.process(audio_path, translate=False)
            hypothesis = result.get("chinese_text", "")
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

    # save ASR result
    output = {
        "dataset": dataset_json,
        "total_samples": len(samples),
        "processed_samples": len([h for h in hypotheses if h != ""]),
        "per_sample": per_sample_results,
    }
    output_file = os.path.join(output_dir, "ASR_result.json")
    save_results_to_json(output, output_file)

    # 4. Evaluate
    evaluator = ASREvaluator()
    eval_results = evaluator.evaluate_lists(refs, hypotheses)

    # 5. Build final output
    final_output = {
        "dataset": dataset_json,
        "total_samples": len(samples),
        "processed_samples": len([h for h in hypotheses if h != ""]),
        "overall_metrics": eval_results["overall"],
        "per_sample": per_sample_results,
        "raw_evaluation": eval_results
    }

    # 6. Save results
    summary_file = os.path.join(output_dir, "test_summary.json")
    save_results_to_json(final_output, summary_file)

    # Print summary
    evaluator.print_summary()

    return final_output

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ASR Integration Test")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON file")
    parser.add_argument("--output_dir", default="test_outputs", help="Output directory")
    parser.add_argument("--asr_device", default="cuda:0", help="Device for ASR")
    parser.add_argument("--asr_mode", default="single", choices=["single", "speaker"])
    args = parser.parse_args()

    run_integration_test(
        dataset_json=args.dataset,
        output_dir=args.output_dir,
        asr_device=args.asr_device,
        asr_mode=args.asr_mode,
    )
