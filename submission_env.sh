#!/usr/bin/env bash

SUBMISSION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LMM_SUBMISSION_ROOT="$SUBMISSION_ROOT"
export LMM_MODEL_ROOT="$LMM_SUBMISSION_ROOT/models"

export QWEN3_ASR_MODEL_PATH="$LMM_MODEL_ROOT/Qwen3-ASR-1.7B"
export TRANSLATION_MODEL_PATH="$LMM_MODEL_ROOT/Qwen2.5-3B-Instruct"
export DEEPFILTER_MODEL_PATH="$LMM_MODEL_ROOT/pretrained/DeepFilterNet3"
export MODERNBERT_MODEL_PATH="$LMM_MODEL_ROOT/modernbert-drive-command-compositional"
export YOLOP_ROOT="$LMM_MODEL_ROOT/external/YOLOP"
export YOLO11_CARLA_MODEL_PATH="$LMM_MODEL_ROOT/scene_understanding/yolo11s_specialized_carla_v1/weights/best.pt"
export VLA_MODEL_PATH="$LMM_MODEL_ROOT/lightweight_vla_adapter/v10/model.pt"
export MODERNBERT_PRETRAINED_PATH="$LMM_MODEL_ROOT/pretrained/ModernBERT-base"
export YOLO11_PRETRAINED_PATH="$LMM_MODEL_ROOT/pretrained/yolo11s.pt"

printf 'LMM_SUBMISSION_ROOT=%s\n' "$LMM_SUBMISSION_ROOT"
printf 'LMM_MODEL_ROOT=%s\n' "$LMM_MODEL_ROOT"
