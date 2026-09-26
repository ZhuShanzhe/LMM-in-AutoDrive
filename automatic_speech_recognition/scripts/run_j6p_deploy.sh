#!/usr/bin/env bash
# J6P deployment chain: float ONNX -> .hbm compile -> alignment check -> board push.
# Run on the host PC (Linux) with the Horizon OE toolchain installed:
#   bash scripts/run_j6p_deploy.sh [--force] [--verify] [--board user@host]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

COMPRESSION_CONFIG="configs/asr/compression.yaml"
DEPLOY_CONFIG="configs/asr/deployment.yaml"
FLOAT_ONNX="outputs/compression/onnx/asr_encoder.onnx"
HBM_DIR="outputs/deployment/j6p"

FORCE=0
VERIFY=0
BOARD=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1 ;;
    --verify) VERIFY=1 ;;
    --board)
      [[ $# -ge 2 ]] || { echo "--board requires user@host" >&2; exit 2; }
      BOARD="$2"; shift ;;
    -h|--help)
      echo "usage: bash scripts/run_j6p_deploy.sh [--force] [--verify] [--board user@host]"
      echo "  --force   re-export the float ONNX even if it already exists"
      echo "  --verify  run the float-vs-hbm alignment check (needs a BPU runtime, i.e. on the board)"
      echo "  --board   scp the compiled .hbm to the J6P target and print the sanity command"
      exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

if [[ $FORCE -eq 1 || ! -f "$FLOAT_ONNX" ]]; then
  echo "[1/4] export float32 ONNX (compression pipeline)"
  python -m src.asr.compression.quantize --config "$COMPRESSION_CONFIG"
else
  echo "[1/4] skipped, reusing $FLOAT_ONNX"
fi

VERIFY_FLAG=""
if [[ $VERIFY -eq 1 ]]; then
  VERIFY_FLAG="--verify"
fi
echo "[2/4] compile float ONNX to J6P .hbm (PTQ calibration + hb_compile)"
python -m src.asr.deployment.hb_convert --config "$DEPLOY_CONFIG" $VERIFY_FLAG

echo "[3/4] artifacts"
ls -l "$HBM_DIR"

if [[ -n "$BOARD" ]]; then
  echo "[4/4] push .hbm to $BOARD:/userdata/asr/"
  if compgen -G "$HBM_DIR/*.hbm" > /dev/null; then
    ssh "$BOARD" "mkdir -p /userdata/asr"
    scp "$HBM_DIR"/*.hbm "$BOARD:/userdata/asr/"
    echo "on board, sanity check with: hrt_model_exec model_info --model_file=/userdata/asr/<name>.hbm"
  else
    echo "no .hbm produced under $HBM_DIR; check the hb_compile log" >&2
    exit 1
  fi
else
  echo "[4/4] skipped board push (pass --board user@host to deploy)"
fi

echo "done. run on board with backend=j6p and hbm_path in $DEPLOY_CONFIG"