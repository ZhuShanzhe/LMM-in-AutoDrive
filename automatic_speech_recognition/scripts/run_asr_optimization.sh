#!/usr/bin/env bash
# ASR optimization chain (standard + dialect + noise datasets).
# Steps: standard eval -> dialect eval -> noise eval -> export/quantize -> quantized accuracy regression.
# Run from the repo root: bash scripts/run_asr_optimization.sh [--force]
#   --force   rerun every step even if previous results are found
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

STANDARD_MAP="data/wav_files/standard/mapping.json"
DIALECT_MAP="data/wav_files/dialect/mapping.json"
NOISE_MAP="data/wav_files/noise/standard_noise/mapping.json"

STANDARD_DIR="outputs/tests/standard"
DIALECT_DIR="outputs/tests/dialect"
NOISE_DIR="outputs/tests/noise"
CONFIG="configs/asr/compression.yaml"

FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    -h|--help)
      echo "usage: bash scripts/run_asr_optimization.sh [--force]"
      echo "  --force   rerun every step even if previous results are found"
      exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

run_step() {  # run_step <marker> <command...>
  local marker="$1"; shift
  local completed_marker
  completed_marker="$(dirname "$marker")/details.json"
  if [[ $FORCE -eq 0 && -f "$marker" && -f "$completed_marker" ]]; then
    echo "  skipped, reusing $marker"
    return 0
  fi
  "$@"
}

echo "[1/4] baseline accuracy (standard dataset)"
run_step "$STANDARD_DIR/summary.json" \
  python tests/asr_test.py --dataset "$STANDARD_MAP" --save-dir "$STANDARD_DIR"

echo "[2/4] robustness accuracy (dialect dataset)"
run_step "$DIALECT_DIR/summary.json" \
  python tests/dialect_test.py --dataset "$DIALECT_MAP" --save-dir "$DIALECT_DIR"

echo "[3/4] robustness accuracy (noise dataset)"
run_step "$NOISE_DIR/summary.json" \
  python tests/noise_test.py --dataset "$NOISE_MAP" --save-dir "$NOISE_DIR"

echo "[4/4] export ONNX + quantize + accuracy regression"
python -m src.asr.compression.quantize --config "$CONFIG" --verify

echo "done. standard: $STANDARD_DIR/summary.json | dialect: $DIALECT_DIR/summary.json | noise: $NOISE_DIR/summary.json | compression: outputs/compression/"