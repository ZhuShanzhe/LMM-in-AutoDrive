#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
if [[ $# -lt 2 || -z "${MODEL_ROOT:-}" ]]; then
  echo 'Challenge only: MODEL_ROOT=/path/to/models bash scripts/run_universal_vla.sh scene1 /fresh/output [launcher options]' >&2
  exit 2
fi
SCENE="$1"
OUTPUT="$2"
shift 2
exec "${PYTHON_BIN:-python}" "$REPO_ROOT/experiment/CARLA/tools/run_challenge_x86.py" \
  --scene "$SCENE" --model-root "$MODEL_ROOT" --output "$OUTPUT" \
  --device "${VLA_DEVICE:-cpu}" --execute "$@"
