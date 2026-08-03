import json
import os
import random
import logging
import shutil
from src import ChatTTSService


class Config:
    def __init__(self):
        self.json_file = "data/translated_commands.json"
        self.model_path = "./models/rvcmd_linux_amd64"
        self.output_dir = "data/wav_files_noise"

        self.num_samples = 500
        self.random_seed = 42

        self.batch_size = 1
        self.compile_model = False
        self.noise_enabled = True
        self.noise_type = "white"
        self.noise_level = 0.01
        self.noise_file = None

        self.log_level = logging.INFO


def main():
    config = Config()
    logging.basicConfig(
        level=config.log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    random.seed(config.random_seed)

    if not os.path.exists(config.json_file):
        logging.error(f"Input file not found: {config.json_file}")
        return

    with open(config.json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    valid_entries = [item for item in data if item.get("translation", "").strip()]
    total_available = len(valid_entries)
    if total_available == 0:
        logging.error("No valid 'translation' fields found in JSON.")
        return

    if total_available < config.num_samples:
        logging.warning(
            f"Only {total_available} entries available, using all instead of requested {config.num_samples}"
        )
        sampled_entries = valid_entries
    else:
        sampled_entries = random.sample(valid_entries, config.num_samples)

    texts = [item["translation"].strip() for item in sampled_entries]
    logging.info(f"Sampled {len(texts)} texts from {config.json_file}")

    service = ChatTTSService(
        model_path=config.model_path,
        compile_model=config.compile_model,
        batch_size=config.batch_size,
        output_dir=config.output_dir,
        noise_enabled=config.noise_enabled,
        noise_type=config.noise_type,
        noise_level=config.noise_level,
        noise_file=config.noise_file,
    )

    service.reset_counter(0)

    logging.info(f"Starting synthesis for {len(texts)} texts with batch_size={config.batch_size}...")
    filepaths = service.synthesize(
        text=texts,
        batch_size=config.batch_size,
        add_noise=config.noise_enabled,
        noise_type=config.noise_type,
        noise_level=config.noise_level,
        noise_file=config.noise_file,
    )

    mapping = []
    renamed_paths = []
    for idx, (item, old_path) in enumerate(zip(sampled_entries, filepaths), start=1):
        new_filename = f"commands_{idx}_noise.wav"
        new_path = os.path.join(config.output_dir, new_filename)

        if os.path.exists(old_path):
            shutil.move(old_path, new_path)
            logging.info(f"Renamed {os.path.basename(old_path)} -> {new_filename}")
        else:
            logging.warning(f"File not found: {old_path}")
            new_path = old_path

        renamed_paths.append(new_path)

        mapping.append({
            "index": idx,
            "original": item.get("original", ""),
            "translation": item.get("translation", ""),
            "audio_file": new_path
        })

    mapping_file = os.path.join(config.output_dir, "file_mapping_noise.json")
    os.makedirs(config.output_dir, exist_ok=True)
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    logging.info(f"File mapping saved to: {mapping_file}")

    logging.info(f"Dataset generation completed. {len(renamed_paths)} files generated.")


if __name__ == "__main__":
    main()
