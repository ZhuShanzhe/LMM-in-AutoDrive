import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import ASRPipeline

def demo_from_config() -> None:
    pipe = ASRPipeline.from_yaml("configs/asr/pipeline.yaml")
    audio = sys.argv[1] if len(sys.argv) > 1 else "data/wav_files/standard/std_00001.wav"

    result = pipe.process(audio, output_json="outputs/pipeline_result.json")
    print("[pipeline] chinese   :", result["text"])
    print("[pipeline] english   :", result["translation"])
    print("[pipeline] output    :", result["output_text"])
    print("[pipeline] optimized :", result["frontend_applied"],
          "| dialect:", result["dialect_normalized"])

def demo_explicit_params() -> None:
    pipe = ASRPipeline(
        asr_model_path="models/Qwen3-ASR-1.7B",
        enable_optimization=True,
        optimization_config="configs/asr/optimization.yaml",
        enable_translation=True,
        translator_model_path="models/Qwen3-1.7B",
        output_language="both",           # chinese | english | both
        output_dir="outputs",
    )
    audio = sys.argv[1] if len(sys.argv) > 1 else "data/wav_files/standard/std_00001.wav"

    records = pipe.process_batch(
        [audio],
        output_json="outputs/pipeline_batch.json",
        output_language="english",
    )
    for r in records:
        print("[batch]", r["audio_file"], "=>", r["output_text"])

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    demo_from_config()
    demo_explicit_params()

if __name__ == "__main__":
    main()