import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.asr import Qwen3ASRService
from src.utils import log_and_print, resolve_devices, setup_logging
from tests.utils.data_loader import load_dataset, subset_by_condition
from tests.utils.evaluator import ASREvaluator, report

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dialect ASR dataset test")
    parser.add_argument("--dataset", default="data/wav_files/dialect/mapping.json")
    parser.add_argument("--commands-file", default=None)
    parser.add_argument("--dialect", default=None, help="keep only records whose 'dialect' equals this value")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="models/Qwen3-ASR-1.7B")
    parser.add_argument("--device", default=None, help="pin every worker to this device; overrides --num-gpus")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of GPUs to use; 1 degenerates to cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--save-dir", default="outputs/tests/dialect")
    parser.add_argument("--progress-every", type=int, default=10, help="print a progress line every N records")
    parser.add_argument("--save-every", type=int, default=50, help="refresh summary.json every N records")
    parser.add_argument("--no-progress", action="store_true", help="disable realtime progress output")
    parser.add_argument("--enable-slots", action="store_true", help="also report optional slot diagnostics")
    parser.add_argument("--log-file", default="logs/tests/dialect_test.log", help="run log path; empty string disables it")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)

    records = load_dataset(args.dataset, args.commands_file, None)
    if args.dialect:
        records = subset_by_condition(records, args.dialect, keys=("dialect",))
        if not records:
            raise SystemExit(f"no records with dialect '{args.dialect}' in {args.dataset}")
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise SystemExit(f"no records loaded from {args.dataset}")
    log_and_print(f"[dialect_test] loaded {len(records)} records")

    devices = [args.device] if args.device else resolve_devices(args.num_gpus)
    services = [Qwen3ASRService(model_id_or_path=args.model, device=d, dtype=args.dtype) for d in devices]
    log_and_print(f"[dialect_test] devices: {devices}")

    save_dir = ROOT / args.save_dir
    lanes = [lambda p, s=svc: s.transcribe_file(p, use_frontend=False) for svc in services]
    summary = ASREvaluator(lanes[0], records, enable_slots=args.enable_slots, transcribes=lanes).run(
        save_dir=str(save_dir),
        label="dialect_test",
        progress=not args.no_progress,
        progress_every=max(1, args.progress_every),
        save_every=max(1, args.save_every),
        workers=len(devices),
    )

    log_and_print(report(summary, "Dialect ASR Test"))

if __name__ == "__main__":
    main()