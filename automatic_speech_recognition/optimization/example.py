import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization import AudioAugmenter, AugmentConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    input_file = "data/example.wav"
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        logger.info("Please place an example.wav in the data/ directory.")
        return

    config = AugmentConfig(
        noise_type="white",          # Type of noise: "white", "pink", "vehicle", "from_file"
        snr_db=15.0,                 # Signal-to-noise ratio in dB
        sample_rate=16000,           # Target sample rate (resamples if needed)
        output_dir="data",           # Output directory
        seed=42,                     # For reproducibility
    )

    augmenter = AudioAugmenter(config)

    output_path = augmenter.process_file(
        input_path=input_file,
        snr_db=15.0,
    )

    logger.info(f"Successfully generated noisy file: {output_path}")


if __name__ == "__main__":
    main()
