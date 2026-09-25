import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from src.asr import Qwen3ASRService
from src.asr.optimization import build_optimizer

def demo_basic() -> None:
    service = Qwen3ASRService(model_id_or_path="models/Qwen3-ASR-1.7B")

    rec = service.transcribe_file(
        "data/wav_files/standard/std_00001.wav",
        output_json="outputs/asr_single.json",
    )
    print("[single]", rec["text"], "| lang:", rec["language"],
          "| time:", rec["processing_time_seconds"])

    paths = [
        "data/wav_files/standard/std_00001.wav",
        "data/wav_files/standard/std_00002.wav",
    ]
    records = service.transcribe_batch(paths, output_json="outputs/asr_batch.json")
    for r in records:
        print("[batch]", r["audio_file"], "=>", r["text"])

def demo_optimized() -> None:
    opt_cfg = yaml.safe_load(
        open("configs/asr/optimization.yaml", encoding="utf-8"))
    optimizer = build_optimizer(opt_cfg)
    frontend, dialect_normalizer = optimizer.as_hooks()

    service = Qwen3ASRService.from_yaml(
        "configs/asr/qwen3_asr.yaml",
        frontend=frontend,
        dialect_normalizer=dialect_normalizer,
    )

    rec = service.transcribe_file(
        "data/wav_files/dialect/sichuan_00001.wav",
        output_json="outputs/asr_optimized.json",
        save_enhanced="outputs/enhanced_00001.wav",
    )
    print("[optimized]", rec["text"],
          "| frontend:", rec["frontend_applied"],
          "| dialect:", rec["dialect_normalized"])

def main() -> None:
    config_path = ROOT / "configs/asr/qwen3_asr.yaml"
    if not config_path.exists():
        print("[info] config not found:", config_path)
    demo_basic()
    demo_optimized()

if __name__ == "__main__":
    main()