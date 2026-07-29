import os
import json
import argparse
import logging
from typing import Dict, Any, Optional, List

from utils.data_loader import load_test_json, save_results_to_json
from optimization import DenoiseService, DenoiserConfig
from qwen_test import run_integration_test

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def detect_id_key(data: List[Dict]) -> str:
    """
    Detect the most likely key used as sample ID from a dataset.

    Tries common ID field names and returns the first one that exists
    and is not None for all entries. If none found, raises ValueError.
    """
    if not data:
        raise ValueError("Dataset is empty.")
    candidate_keys = ["index", "id", "idx", "sample_id", "uid", "ID"]
    first_item = data[0]
    for key in candidate_keys:
        if key in first_item and first_item[key] is not None:
            logger.info(f"Using '{key}' as sample ID key.")
            return key
    logger.error(f"Available keys in the first item: {list(first_item.keys())}")
    raise ValueError(
        "Could not find a suitable ID field. Please ensure your dataset contains an ID column "
        "(e.g., 'index', 'id', etc.) and specify it with the '--id_key' argument if needed."
    )


def create_clean_subset_from_noisy(
    noisy_dataset_json: str,
    original_dataset_json: str,
    output_dir: str,
    audio_key: str = "audio_file",
    ref_key: str = "translation",
) -> str:
    """
    Extract clean audio paths from the original dataset using a suitable ID field.

    This function automatically detects the ID field from the noisy dataset
    (e.g., 'index', 'id', etc.) and matches with the original dataset.

    Args:
        noisy_dataset_json: Path to noisy dataset JSON.
        original_dataset_json: Path to the original full dataset JSON.
        output_dir: Directory to save the generated clean dataset JSON.
        audio_key: Key in original dataset for audio file path.
        ref_key: Key for reference text in original dataset.

    Returns:
        Path to the generated clean dataset JSON.
    """
    noisy_data = load_test_json(noisy_dataset_json)
    original_data = load_test_json(original_dataset_json)

    id_key = detect_id_key(noisy_data)

    original_map = {str(item.get(id_key)): item for item in original_data if item.get(id_key) is not None}

    clean_data: List[Dict[str, Any]] = []
    for noisy_item in noisy_data:
        idx = str(noisy_item.get(id_key))
        if idx == "None" or idx not in original_map:
            logger.debug(f"ID {idx} not found in original dataset, skipping.")
            continue

        orig_item = original_map[idx]
        clean_item = {
            "id": idx,
            "index": idx,
            "original": orig_item.get("original", ""),
            "reference": orig_item.get("reference", ""),
            audio_key: orig_item.get(audio_key),
        }
        for k in ["original", "reference"]:
            if k in orig_item:
                clean_item[k] = orig_item[k]
        clean_data.append(clean_item)

    clean_json_path = os.path.join(output_dir, "clean_from_noisy.json")
    save_results_to_json(clean_data, clean_json_path)
    logger.info(f"Created clean subset with {len(clean_data)} samples (matched from {len(noisy_data)} noisy entries): {clean_json_path}")
    return clean_json_path


def apply_denoising_to_dataset(
    dataset_json: str,
    denoiser: DenoiseService,
    output_dir: str,
    audio_key: str = "audio_file",
) -> str:
    """
    Apply denoising to all audio files in the dataset and create a new JSON
    mapping that points to the denoised files.

    Outputs:
        - WAV files in {output_dir}/denoisy/
        - Mapping JSON at {output_dir}/denoisy_mapping.json

    Args:
        dataset_json: Path to dataset JSON (pointing to noisy audio).
        denoiser: DenoiseService instance.
        output_dir: Root output directory.
        audio_key: JSON key containing audio file path.

    Returns:
        Path to new JSON mapping pointing to denoised files.
    """
    denoisy_dir = os.path.join(output_dir, "denoisy")
    os.makedirs(denoisy_dir, exist_ok=True)

    data = load_test_json(dataset_json)
    new_data = []

    for item in data:
        orig_path = item.get(audio_key)
        if not orig_path or not os.path.exists(orig_path):
            logger.warning(f"Skipping {orig_path}: file not found")
            new_item = item.copy()
            new_item[audio_key] = orig_path
            new_data.append(new_item)
            continue

        base = os.path.splitext(os.path.basename(orig_path))[0]
        denoised_path = os.path.join(denoisy_dir, f"{base}_denoised.wav")

        try:
            denoiser.denoise(
                audio=orig_path,
                output_path=denoised_path,
                output_sr=denoiser.config.output_sr,
            )
            new_item = item.copy()
            new_item[audio_key] = denoised_path
            new_data.append(new_item)
            logger.info(f"Denoised: {orig_path} -> {denoised_path}")
        except Exception as e:
            logger.error(f"Denoising failed for {orig_path}: {e}")
            new_item = item.copy()
            new_item[audio_key] = orig_path
            new_data.append(new_item)

    new_json = os.path.join(output_dir, "denoisy_mapping.json")
    save_results_to_json(new_data, new_json)
    return new_json


def compare_metrics(
    results_noisy: Dict[str, Any],
    results_denoised: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    """
    Compare metrics from two evaluation runs and generate a comparison report.

    Args:
        results_noisy: Evaluation result dict from noisy dataset (contains 'overall_metrics').
        results_denoised: Evaluation result dict from denoised dataset.
        output_dir: Directory to save comparison JSON.

    Returns:
        Comparison dictionary.
    """
    noisy_metrics = results_noisy["overall_metrics"]
    denoised_metrics = results_denoised["overall_metrics"]

    comparison = {
        "total_samples": noisy_metrics["total_samples"],
        "noisy": {
            "average_cer": noisy_metrics["average_cer"],
            "average_wer": noisy_metrics["average_wer"],
            "sentence_accuracy_rate": noisy_metrics["sentence_accuracy_rate"],
        },
        "denoised": {
            "average_cer": denoised_metrics["average_cer"],
            "average_wer": denoised_metrics["average_wer"],
            "sentence_accuracy_rate": denoised_metrics["sentence_accuracy_rate"],
        },
    }

    comp_file = os.path.join(output_dir, "comparison.json")
    save_results_to_json(comparison, comp_file)
    logger.info(f"Comparison report saved to {comp_file}")

    print("\n" + "=" * 70)
    print("Denoising Effect on ASR Performance (Noisy vs Denoised)")
    print("=" * 70)
    print(f"Total samples: {comparison['total_samples']}")
    print("\n{:<20} {:<15} {:<15} {:<15}".format("Metric", "Noisy", "Denoised", "Change"))
    print("=" * 70 + "\n")

    return comparison


def run_denoising_evaluation(
    dataset_json: str,
    original_dataset_json: Optional[str] = None,
    clean_dataset_json: Optional[str] = None,
    output_dir: str = "test_results_denoising",
    asr_device: str = "cuda:0",
    load_type: str = "local",
    model_path: str = "./models/Qwen3-ASR-1.7B",
    language: str = "Chinese",
    denoiser_config: Optional[Dict] = None,
    skip_noisy: bool = False,
) -> Dict[str, Any]:
    """
    Main function to evaluate denoising effect on ASR.

    Args:
        dataset_json: Path to noisy dataset JSON.
        original_dataset_json: Path to original full dataset JSON (needed to extract clean subset).
        clean_dataset_json: Optional pre-built clean dataset JSON (overrides automatic extraction).
        output_dir: Directory to store all results.
        asr_device: Device for ASR.
        load_type: 'custom' or 'local' for ASR model loading.
        model_path: Path to local Qwen3-ASR model.
        language: Language hint for ASR.
        denoiser_config: Dict to override DenoiserConfig.
        skip_noisy: If True, skip ASR on noisy data (use existing results).

    Returns:
        Comparison dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)

    if denoiser_config is None:
        denoiser_config = {}
    d_config = DenoiserConfig(**denoiser_config)
    denoiser = DenoiseService(config=d_config)

    denoisy_json = apply_denoising_to_dataset(
        dataset_json=dataset_json,
        denoiser=denoiser,
        output_dir=output_dir,
    )

    clean_json = clean_dataset_json
    if clean_json is None and original_dataset_json is not None:
        clean_json = create_clean_subset_from_noisy(
            noisy_dataset_json=dataset_json,
            original_dataset_json=original_dataset_json,
            output_dir=output_dir,
        )

    clean_metrics = None
    if clean_json:
        clean_output_dir = os.path.join(output_dir, "clean_asr")
        results_clean = run_integration_test(
            dataset_json=clean_json,
            output_dir=clean_output_dir,
            asr_device=asr_device,
            load_type=load_type,
            model_path=model_path,
            language=language,
            enable_enhancement=False,
            enable_dialect_mapping=False,
            ref_key="reference",
        )
        clean_metrics = results_clean["overall_metrics"]

        clean_baseline_file = os.path.join(output_dir, "clean_baseline.json")
        save_results_to_json(clean_metrics, clean_baseline_file)
        logger.info(f"Clean baseline metrics saved to {clean_baseline_file}")

        print("\n" + "=" * 70)
        print("Clean Audio ASR Performance (Baseline)")
        print("=" * 70)
        print(f"Total samples: {clean_metrics['total_samples']}")
        print(f"Average CER: {clean_metrics['average_cer']:.4f}")
        print(f"Average WER: {clean_metrics['average_wer']:.4f}")
        print(f"Sentence Accuracy: {clean_metrics['sentence_accuracy_rate']:.2%}")
        print("=" * 70 + "\n")

    if not skip_noisy:
        noisy_output_dir = os.path.join(output_dir, "noisy_asr")
        results_noisy = run_integration_test(
            dataset_json=dataset_json,
            output_dir=noisy_output_dir,
            asr_device=asr_device,
            load_type=load_type,
            model_path=model_path,
            language=language,
            enable_enhancement=False,
            enable_dialect_mapping=False,
            ref_key="translation",
        )
    else:
        noisy_summary = os.path.join(output_dir, "noisy_asr", "test_summary_qwen3.json")
        if not os.path.exists(noisy_summary):
            raise FileNotFoundError(
                f"Noisy ASR results not found at {noisy_summary} and skip_noisy is True."
            )
        with open(noisy_summary, 'r') as f:
            results_noisy = json.load(f)

    denoised_output_dir = os.path.join(output_dir, "denoised_asr")
    results_denoised = run_integration_test(
        dataset_json=denoisy_json,
        output_dir=denoised_output_dir,
        asr_device=asr_device,
        load_type=load_type,
        model_path=model_path,
        language=language,
        enable_enhancement=False,
        enable_dialect_mapping=False,
        ref_key="reference",
    )

    comparison = compare_metrics(results_noisy, results_denoised, output_dir)

    if clean_metrics:
        comparison["clean_baseline"] = clean_metrics
        comp_file = os.path.join(output_dir, "comparison.json")
        with open(comp_file, 'r') as f:
            existing = json.load(f)
        existing["clean_baseline"] = clean_metrics
        save_results_to_json(existing, comp_file)

    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate denoising effect on ASR, with optional clean baseline (printed separately)."
    )
    parser.add_argument("--dataset", required=True, help="Path to noisy dataset JSON.")
    parser.add_argument("--original_dataset",
                        help="Path to the original full dataset JSON (e.g., file_mapping.json). Required to extract clean baseline.")
    parser.add_argument("--clean_dataset", help="Path to pre-built clean dataset JSON (optional, overrides automatic extraction).")
    parser.add_argument("--output_dir", default="test_results_denoising", help="Output directory.")
    parser.add_argument("--asr_device", default="cuda:0", help="Device for ASR (cuda:0 / cpu).")
    parser.add_argument("--load_type", default="local", choices=["custom", "local"],
                        help="ASR model load type.")
    parser.add_argument("--model_path", default="./models/Qwen3-ASR-1.7B",
                        help="Path to local Qwen3-ASR model.")
    parser.add_argument("--language", default="Chinese", help="Language hint for ASR.")
    parser.add_argument("--denoiser_model", default="DeepFilterNet3", help="Denoiser model name.")
    parser.add_argument("--denoiser_output_sr", type=int, default=16000,
                        help="Denoiser output sample rate.")
    parser.add_argument("--skip_noisy", action="store_true",
                        help="Skip ASR on noisy data (use existing results).")
    args = parser.parse_args()

    denoiser_config = {
        "model_name": args.denoiser_model,
        "output_sr": args.denoiser_output_sr,
    }

    run_denoising_evaluation(
        dataset_json=args.dataset,
        original_dataset_json=args.original_dataset,
        clean_dataset_json=args.clean_dataset,
        output_dir=args.output_dir,
        asr_device=args.asr_device,
        load_type=args.load_type,
        model_path=args.model_path,
        language=args.language,
        denoiser_config=denoiser_config,
        skip_noisy=args.skip_noisy,
    )
