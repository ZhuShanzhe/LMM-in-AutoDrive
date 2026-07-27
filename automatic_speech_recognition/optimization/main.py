import os
import json
import argparse
from tqdm import tqdm
from config import Config
from tts_engine import TtsEngine
from audio_processor import AudioProcessor


def synthesize_with_dialect(
    text: str,
    dialect: str,
    config: Config,
    speaker: str = None,
    add_noise: bool = True,
) -> str:
    """
    Synthesize speech with dialect/emotion control and optional noise.

    Args:
        text: Text to synthesize.
        dialect: Instruction for dialect/emotion.
        config: Configuration object.
        speaker: Voice speaker.
        add_noise: Whether to add background noise.

    Returns:
        Path to the final audio file.
    """
    os.makedirs(config.output_dir, exist_ok=True)

    engine = TtsEngine(config)
    clean_path = os.path.join(config.output_dir, "temp_clean.wav")

    audio_path = engine.synthesize(
        text=text,
        instruct_text=dialect,
        speaker=speaker,
        output_path=clean_path,
    )

    if add_noise and config.noise_enabled:
        processor = AudioProcessor(config)
        noisy_path = os.path.join(config.output_dir, "output_noisy.wav")
        output_path = processor.add_background_noise(
            audio_path,
            output_path=noisy_path,
        )
        if os.path.exists(clean_path):
            os.remove(clean_path)
    else:
        output_path = os.path.join(config.output_dir, "output.wav")
        if os.path.exists(clean_path) and clean_path != output_path:
            os.rename(clean_path, output_path)

    return output_path


def batch_synthesize(
    texts: list,
    dialects: list,
    config: Config,
    speaker: str = None,
    add_noise: bool = True,
) -> list:
    """
    Batch synthesize multiple texts.

    Args:
        texts: List of texts.
        dialects: List of dialect instructions (same length).
        config: Configuration object.
        speaker: Voice speaker.
        add_noise: Whether to add noise.

    Returns:
        List of output file paths.
    """
    os.makedirs(config.output_dir, exist_ok=True)

    if dialects is None:
        dialects = ["speak in standard Mandarin"] * len(texts)
    elif len(dialects) != len(texts):
        raise ValueError("Length of 'texts' and 'dialects' must match.")

    results = []
    for i, (text, dialect) in enumerate(tqdm(zip(texts, dialects), total=len(texts))):
        output_path = os.path.join(config.output_dir, f"command_{i+1:04d}.wav")
        result = synthesize_with_dialect(
            text=text,
            dialect=dialect,
            config=config,
            speaker=speaker,
            add_noise=add_noise,
        )
        if os.path.exists(result) and result != output_path:
            os.rename(result, output_path)
        results.append(output_path)

    return results


def main():
    parser = argparse.ArgumentParser(description="CosyVoice dialect synthesis with noise augmentation.")
    parser.add_argument("--text", type=str, help="Text to synthesize (single mode).")
    parser.add_argument("--dialect", type=str, default="speak in standard Mandarin", help="Dialect/emotion instruction.")
    parser.add_argument("--speaker", type=str, help="Speaker voice (overrides config).")
    parser.add_argument("--output_dir", type=str, help="Output directory (overrides config).")
    parser.add_argument("--no_noise", action="store_true", help="Disable background noise addition.")
    parser.add_argument("--batch", action="store_true", help="Batch mode (requires --input_json).")
    parser.add_argument("--input_json", type=str, help="JSON file with list of {'text': ..., 'dialect': ...} for batch.")
    parser.add_argument("--config_json", type=str, help="JSON file to load configuration from.")

    args = parser.parse_args()

    # Load configuration
    if args.config_json:
        with open(args.config_json, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        config = Config.from_dict(config_dict)
    else:
        config = Config.from_env()   # Load from environment variables (falls back to defaults)

    # Override with command-line arguments
    if args.output_dir:
        config.output_dir = args.output_dir

    add_noise = not args.no_noise

    if args.batch:
        if not args.input_json:
            print("Error: --input_json is required for batch mode.")
            return

        with open(args.input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        texts = [item['text'] for item in data]
        dialects = [item.get('dialect', 'speak in standard Mandarin') for item in data]

        results = batch_synthesize(
            texts=texts,
            dialects=dialects,
            config=config,
            speaker=args.speaker,
            add_noise=add_noise,
        )
        print(f"Batch synthesis completed. {len(results)} files generated.")
    else:
        if not args.text:
            print("Error: --text is required for single mode.")
            return

        output_path = synthesize_with_dialect(
            text=args.text,
            dialect=args.dialect,
            config=config,
            speaker=args.speaker,
            add_noise=add_noise,
        )
        print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()
