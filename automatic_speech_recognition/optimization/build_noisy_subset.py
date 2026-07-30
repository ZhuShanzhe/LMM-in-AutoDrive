import os
import json
import argparse
import random
import logging
from typing import List, Dict, Any, Optional, Tuple, Set

from optimization import AudioAugmenter, AugmentConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_dataset(json_path: str) -> List[Dict[str, Any]]:
    """Load the dataset from a JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_dataset(data: List[Dict[str, Any]], json_path: str) -> None:
    """Save dataset to a JSON file."""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_asr_results(asr_result_path: str) -> List[Dict[str, Any]]:
    """
    Load ASR results JSON and extract the per_sample list.
    Expected format: {"per_sample": [...]} or directly a list.
    """
    data = load_dataset(asr_result_path)
    if isinstance(data, dict) and "per_sample" in data:
        return data["per_sample"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("ASR result JSON must contain 'per_sample' list or be a list itself.")


def get_correct_sample_ids(asr_per_sample: List[Dict[str, Any]], id_key: str = "id") -> Set[str]:
    """
    Extract IDs of samples where reference == hypothesis.
    Returns a set of IDs (as strings).
    """
    correct_ids = set()
    for item in asr_per_sample:
        ref = item.get("reference", "")
        hyp = item.get("hypothesis", "")
        if ref == hyp and ref != "":
            # Use the provided id_key, fallback to "index" or "id"
            sample_id = item.get(id_key)
            if sample_id is None:
                # Try alternative keys
                sample_id = item.get("index", item.get("id"))
            if sample_id is not None:
                correct_ids.add(str(sample_id))
            else:
                logger.warning(f"Sample without ID found: {item}")
    return correct_ids


def sample_subset_from_correct(
    data: List[Dict[str, Any]],
    correct_ids: Set[str],
    num_samples: int,
    seed: int,
    id_key: str = "index"
) -> List[Dict[str, Any]]:
    """
    Filter dataset to only include samples whose id is in correct_ids,
    then randomly sample num_samples.
    """
    # Convert data id to string for comparison
    filtered = [item for item in data if str(item.get(id_key, "")) in correct_ids]
    logger.info(f"Found {len(filtered)} samples with correct ASR recognition.")

    if len(filtered) == 0:
        raise ValueError("No samples with correct ASR recognition found. Check ASR result file.")

    if len(filtered) < num_samples:
        logger.warning(f"Only {len(filtered)} correct samples available. Using all instead of {num_samples}.")
        return filtered

    rng = random.Random(seed)
    return rng.sample(filtered, num_samples)


def generate_random_noise_config() -> Tuple[str, float]:
    """
    Generate a random noise configuration.
    """
    noise_types = ["white"]  # You can add 'pink', 'vehicle' if needed
    noise_type = random.choice(noise_types)
    snr_db = round(random.uniform(0.0, 20.0), 1)
    return noise_type, snr_db


def build_noisy_subset(
    dataset_path: str,
    asr_result_path: str,
    output_dir: str,
    num_samples: int = 500,
    seed: int = 42,
    sample_rate: int = 16000,
    audio_key: str = "audio_file",
    id_key: str = "index"
) -> None:
    full_data = load_dataset(dataset_path)
    logger.info(f"Total samples in dataset: {len(full_data)}")

    asr_per_sample = load_asr_results(asr_result_path)
    correct_ids = get_correct_sample_ids(asr_per_sample, id_key=id_key)
    logger.info(f"Number of correctly recognized samples: {len(correct_ids)}")

    sampled_data = sample_subset_from_correct(full_data, correct_ids, num_samples, seed, id_key)
    logger.info(f"Sampled {len(sampled_data)} entries for noise addition.")

    noisy_dir = os.path.join(output_dir, "noisy")
    os.makedirs(noisy_dir, exist_ok=True)

    config = AugmentConfig(
        noise_type="white",
        snr_db=15.0,
        sample_rate=sample_rate,
        output_dir=noisy_dir,
        seed=seed,
    )
    augmenter = AudioAugmenter(config)

    clean_mapping = []
    noisy_mapping = []

    for idx, item in enumerate(sampled_data, 1):
        original_audio = item.get(audio_key)
        if not original_audio or not os.path.exists(original_audio):
            logger.warning(f"Skipping {original_audio}: file not found.")
            continue

        noise_type, snr_db = generate_random_noise_config()
        logger.debug(f"Adding {noise_type} noise, SNR={snr_db:.1f} dB to {os.path.basename(original_audio)}")

        base_name = os.path.splitext(os.path.basename(original_audio))[0]
        noisy_filename = f"{base_name}_noise_{noise_type}_{snr_db:.1f}dB.wav"
        noisy_path = os.path.join(noisy_dir, noisy_filename)

        try:
            augmenter.process_file(
                input_path=original_audio,
                output_path=noisy_path,
                snr_db=snr_db,
                noise_type=noise_type,
            )
        except Exception as e:
            logger.error(f"Failed to process {original_audio}: {e}")
            continue

        clean_item = item.copy()
        clean_item[audio_key] = original_audio
        clean_mapping.append(clean_item)

        noisy_item = item.copy()
        noisy_item[audio_key] = noisy_path
        noisy_item["noise_type"] = noise_type
        noisy_item["snr_db"] = snr_db
        noisy_mapping.append(noisy_item)

        logger.info(f"[{idx}/{len(sampled_data)}] Processed: {os.path.basename(original_audio)} -> {os.path.basename(noisy_path)}")

    noisy_json = os.path.join(output_dir, "noisy_mapping.json")
    save_dataset(noisy_mapping, noisy_json)

    logger.info(f"Processing complete. Generated {len(clean_mapping)} clean and {len(noisy_mapping)} noisy mappings.")
    # logger.info(f"Clean mapping saved to: {clean_json}")
    logger.info(f"Noisy mapping saved to: {noisy_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Sample a subset from correctly recognized audios and add random noise."
    )
    parser.add_argument("--dataset", required=True, help="Path to full dataset JSON file (clean).")
    parser.add_argument("--asr_result", required=True,
                        help="Path to ASR result JSON (contains 'per_sample' list with reference/hypothesis).")
    parser.add_argument("--output_dir", required=True, help="Output root directory.")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples to draw.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Target sample rate in Hz.")
    parser.add_argument("--audio_key", default="audio_file", help="JSON key for audio file path.")
    parser.add_argument("--id_key", default="index", help="JSON key for sample ID (default: index).")
    args = parser.parse_args()

    build_noisy_subset(
        dataset_path=args.dataset,
        asr_result_path=args.asr_result,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        sample_rate=args.sample_rate,
        audio_key=args.audio_key,
        id_key=args.id_key,
    )


if __name__ == "__main__":
    main()
