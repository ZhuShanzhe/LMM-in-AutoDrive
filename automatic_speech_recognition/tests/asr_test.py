import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.asr import Qwen3ASRService
from src.utils import log_and_print, resolve_devices, setup_logging
from tests.utils.data_loader import load_dataset
from tests.utils.evaluator import ASREvaluator, report

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standard ASR dataset test")
    parser.add_argument("--dataset", default="data/wav_files/standard/mapping.json")
    parser.add_argument("--commands-file", default=None, help="required if --dataset is a directory")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="models/Qwen3-ASR-1.7B")
    parser.add_argument("--device", default=None, help="pin every worker to this device; overrides --num-gpus")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of GPUs to use; 1 degenerates to cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--language", default=None)
    parser.add_argument("--save-dir", default="outputs/tests/standard")
    parser.add_argument("--progress-every", type=int, default=10, help="print a progress line every N records")
    parser.add_argument("--save-every", type=int, default=50, help="refresh summary.json every N records")
    parser.add_argument("--no-progress", action="store_true", help="disable realtime progress output")
    parser.add_argument("--enable-slots", action="store_true", help="also report optional slot diagnostics")
    parser.add_argument("--log-file", default="logs/tests/asr_test.log", help="run log path; empty string disables it")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)

    records = load_dataset(args.dataset, args.commands_file, args.limit)
    if not records:
        raise SystemExit(f"no records loaded from {args.dataset}")
    log_and_print(f"[asr_test] loaded {len(records)} records")

    devices = [args.device] if args.device else resolve_devices(args.num_gpus)
    services = [Qwen3ASRService(model_id_or_path=args.model, device=device, dtype=args.dtype, language=args.language) for device in devices]
    log_and_print(f"[asr_test] devices: {devices}")

    evaluator = ASREvaluator(
        services[0].transcribe_file,
        records,
        enable_slots=args.enable_slots,
        transcribes=[service.transcribe_file for service in services],
    )
    summary = evaluator.run(
        save_dir=str(ROOT / args.save_dir),
        progress=not args.no_progress,
        progress_every=max(1, args.progress_every),
        save_every=max(1, args.save_every),
        workers=len(services),
    )
    log_and_print(report(summary, "Standard ASR Test"))

if __name__ == "__main__":
    main()