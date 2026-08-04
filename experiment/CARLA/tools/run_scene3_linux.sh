#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CARLA_EXPERIMENT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
EGO_CONTROLLER="${EGO_CONTROLLER:-route-pid}"
SCENE3_OUTPUT_DIR="${SCENE3_OUTPUT_DIR:-${CARLA_EXPERIMENT_DIR}/outputs/scene3_linux_run}"

exec "${PYTHON_BIN}" "${CARLA_EXPERIMENT_DIR}/run_emergency_response_6km.py" \
  --runtime-config "${CARLA_EXPERIMENT_DIR}/configs/scene_3_emergency_6km_runtime.json" \
  --output-dir "${SCENE3_OUTPUT_DIR}" \
  --ego-controller "${EGO_CONTROLLER}" \
  "$@"
