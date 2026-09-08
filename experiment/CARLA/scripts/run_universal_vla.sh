#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/submission_env.sh" >/dev/null

PYTHON_BIN="${PYTHON_BIN:-python}"
CARLA_HOST="${CARLA_HOST:-127.0.0.1}"
CARLA_PORT="${CARLA_PORT:-2000}"
VLA_DEVICE="${VLA_DEVICE:-cuda}"
VLA_PRECISION="${VLA_PRECISION:-fp16}"
SCENE2_VARIANT="${SCENE2_VARIANT:-0}"
SCENE3_VARIANT="${SCENE3_VARIANT:-auto}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SCENE="${1:-}"
OUTPUT_BASE="${2:-$REPO_ROOT/experiment/CARLA/outputs/submission}"

usage() {
  echo "Usage: $0 <scene1|scene2|scene3> [output-base]" >&2
  exit 2
}

[[ -n "$SCENE" ]] || usage
[[ -f "$VLA_MODEL_PATH" ]] || {
  echo "Missing VLA checkpoint: $VLA_MODEL_PATH" >&2
  exit 2
}
[[ -f "$VLA_CONFIG_PATH" ]] || {
  echo "Missing VLA config: $VLA_CONFIG_PATH" >&2
  exit 2
}
[[ -d "$MODERNBERT_MODEL_PATH" ]] || {
  echo "Missing command parser: $MODERNBERT_MODEL_PATH" >&2
  exit 2
}

mkdir -p "$OUTPUT_BASE"
cd "$REPO_ROOT"

case "$SCENE" in
  scene1)
    OUT="$OUTPUT_BASE/scene1_$RUN_TAG"
    mkdir -p "$OUT"
    exec "$PYTHON_BIN" experiment/CARLA/run_control_experiment.py \
      basic_voice_urban_5km \
      --host "$CARLA_HOST" --port "$CARLA_PORT" \
      --map Town04_Opt \
      --scenario-config experiment/CARLA/configs/basic_voice_urban_5km.json \
      --duration-s 900 \
      --decision-source vla_scene_bridge \
      --command-parser-model "$MODERNBERT_MODEL_PATH" \
      --command-parser-device cpu \
      --vla-checkpoint "$VLA_MODEL_PATH" \
      --vla-config "$VLA_CONFIG_PATH" \
      --vla-device "$VLA_DEVICE" --vla-precision "$VLA_PRECISION" \
      --output-dir "$OUT" \
      --video-output "$OUT/scene1_5km.mp4" \
      --video-fps 20 --camera-view chase --video-overlay
    ;;
  scene2)
    OUT="$OUTPUT_BASE/scene2_variant${SCENE2_VARIANT}_$RUN_TAG"
    mkdir -p "$OUT"
    exec "$PYTHON_BIN" experiment/CARLA/run_complex_avoidance_town05.py \
      --host "$CARLA_HOST" --port "$CARLA_PORT" \
      --duration 0 --competition-run \
      --variant-index "$SCENE2_VARIANT" \
      --output-dir "$OUT" \
      --video-output "$OUT/scene2_8km.mp4" --video-overlay \
      --record-ground-truth --ground-truth-every-n 5 \
      --record-multimodal \
      --vla-checkpoint "$VLA_MODEL_PATH" \
      --vla-config "$VLA_CONFIG_PATH" \
      --command-parser-model "$MODERNBERT_MODEL_PATH" \
      --vla-device "$VLA_DEVICE" --vla-precision "$VLA_PRECISION" \
      --vla-decision-every-n 3
    ;;
  scene3)
    OUT="$OUTPUT_BASE/scene3_$RUN_TAG"
    mkdir -p "$OUT"
    exec "$PYTHON_BIN" experiment/CARLA/run_emergency_response_6km.py \
      --host "$CARLA_HOST" --port "$CARLA_PORT" \
      --duration 0 --require-complete-scene \
      --event-variant "$SCENE3_VARIANT" \
      --output-dir "$OUT" \
      --camera-mode chase-only \
      --presentation-lighting official-rainy-night \
      --camera-width 960 --camera-height 540 \
      --video-output "$OUT/scene3_6km.mp4" \
      --video-fps 20 --video-overlay \
      --record-ground-truth --ground-truth-every-n 5 \
      --ego-controller vla-route-pid \
      --vla-checkpoint "$VLA_MODEL_PATH" \
      --vla-config "$VLA_CONFIG_PATH" \
      --vla-parser-model "$MODERNBERT_MODEL_PATH" \
      --vla-device "$VLA_DEVICE" --vla-precision "$VLA_PRECISION" \
      --vla-decision-every-n 3
    ;;
  *) usage ;;
esac
