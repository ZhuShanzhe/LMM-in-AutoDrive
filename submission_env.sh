#!/usr/bin/env bash

# Source this file from the repository root or from any working directory.
SUBMISSION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LMM_SUBMISSION_ROOT="$SUBMISSION_ROOT"
export LMM_MODEL_ROOT="${MODEL_ROOT:-$LMM_SUBMISSION_ROOT/models}"

export MODERNBERT_MODEL_PATH="${MODERNBERT_MODEL_PATH:-$LMM_MODEL_ROOT/modernbert-drive-command-compositional}"
export VLA_MODEL_PATH="${VLA_MODEL_PATH:-$LMM_MODEL_ROOT/lightweight_vla_adapter/universal_three_scene_v6_sensor_policy/model.pt}"
export VLA_CONFIG_PATH="${VLA_CONFIG_PATH:-$LMM_SUBMISSION_ROOT/lightweight_vla_adapter/configs/universal_three_scene_v6_sensor_policy.json}"
export YOLOP_ROOT="${YOLOP_ROOT:-$LMM_MODEL_ROOT/external/YOLOP}"
export YOLO11_CARLA_MODEL_PATH="${YOLO11_CARLA_MODEL_PATH:-$LMM_MODEL_ROOT/scene_understanding/yolo11s_specialized_carla_v1/weights/best.pt}"

printf 'LMM_SUBMISSION_ROOT=%s\n' "$LMM_SUBMISSION_ROOT"
printf 'LMM_MODEL_ROOT=%s\n' "$LMM_MODEL_ROOT"
printf 'VLA_MODEL_PATH=%s\n' "$VLA_MODEL_PATH"
