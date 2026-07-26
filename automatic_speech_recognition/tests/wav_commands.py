import json
import os
import logging
from tts import ChatTTSService


class Config:
    def __init__(self):
        self.json_file = "data/translated_commands.json"
        self.model_path = "./models/rvcmd_linux_amd64"
        self.output_dir = "data/wav_files"

        self.batch_size = 1
        self.compile_model = True
        self.noise_enabled = False
        self.noise_type = "white"
        self.noise_level = 0.005
        self.noise_file = None

        self.log_level = logging.INFO


def main():
    config = Config()
    logging.basicConfig(
        level=config.log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if not os.path.exists(config.json_file):
        logging.error(f"Input file not found: {config.json_file}")
        return

    with open(config.json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = []
    for item in data:
        translation = item.get("translation", "").strip()
        if translation:
            texts.append(translation)

    if not texts:
        logging.error("No valid 'translation' fields found in JSON.")
        return

    logging.info(f"Loaded {len(texts)} text entries from {config.json_file}")

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

    logging.info(f"Starting synthesis for {len(texts)} texts with batch_size={config.batch_size}...")
    filepaths = service.synthesize(
        text=texts,
        batch_size=config.batch_size,
        add_noise=config.noise_enabled,
        noise_type=config.noise_type,
        noise_level=config.noise_level,
        noise_file=config.noise_file,
    )

    logging.info(f"Synthesis completed. Generated {len(filepaths)} files:")
    for idx, fp in enumerate(filepaths, 1):
        logging.info(f"  {idx:04d}. {fp}")

    mapping = []
    for idx, (item, filepath) in enumerate(zip(data, filepaths), 1):
        mapping.append({
            "index": idx,
            "original": item.get("original", ""),
            "translation": item.get("translation", ""),
            "audio_file": filepath
        })
    mapping_file = os.path.join(config.output_dir, "file_mapping.json")
    os.makedirs(config.output_dir, exist_ok=True)
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    logging.info(f"File mapping saved to: {mapping_file}")


if __name__ == "__main__":
    main()
