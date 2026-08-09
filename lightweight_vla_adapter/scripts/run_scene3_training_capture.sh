#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 OUTPUT_DIR LIGHTING_PROFILE SPEED_KMH SEED" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_RELATIVE="$1"
LIGHTING_PROFILE="$2"
SPEED_KMH="$3"
SEED="$4"

case "${OUTPUT_RELATIVE}" in
  /*|*..*)
    echo "OUTPUT_DIR must be a relative path without '..'" >&2
    exit 2
    ;;
esac
case "${LIGHTING_PROFILE}" in
  official-rainy-night|rainy-daylight|clear-daylight) ;;
  *)
    echo "unsupported lighting profile: ${LIGHTING_PROFILE}" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_RELATIVE}"
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "refusing to overwrite existing output: ${OUTPUT_DIR}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}"

PYTHON_BIN="${SCENE3_PYTHON:-/root/autodl-tmp/conda_envs/command_parser/bin/python}"
set +e
"${PYTHON_BIN}" "${REPO_ROOT}/experiment/CARLA/run_emergency_response_6km.py" \
  --runtime-config "${REPO_ROOT}/experiment/CARLA/configs/scene_3_emergency_6km_runtime.json" \
  --output-dir "${OUTPUT_DIR}" \
  --duration 0 \
  --fixed-delta-seconds 0.05 \
  --camera-tick 1.0 \
  --camera-mode four-view-plus-chase \
  --presentation-lighting "${LIGHTING_PROFILE}" \
  --record-ground-truth \
  --ground-truth-every-n 20 \
  --ego-speed-kmh "${SPEED_KMH}" \
  --ego-controller route-pid \
  --seed "${SEED}" \
  --require-complete-scene \
  >"${OUTPUT_DIR}/runner.log" 2>&1
STATUS=$?
set -e
printf '%s\n' "${STATUS}" >"${OUTPUT_DIR}/exit_code.txt"
exit "${STATUS}"
