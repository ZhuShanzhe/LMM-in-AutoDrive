#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate drivelm_carla

cd /root/autodl-tmp/LMM-in-AutoDrive/experiments/drivelm/external/DriveLM-CARLA

python - <<'PY'
import torch
import carla
import numpy
import cv2
import pygame
import shapely
import py_trees
import requests
import tqdm

print("imports_ok")
print("torch:", torch.__version__)
print("torch_cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    x = torch.randn(512, 512, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("gpu_matmul_ok:", tuple(y.shape), y.dtype)
print("numpy:", numpy.__version__)
print("cv2:", cv2.__version__)
PY

python vqa_dataset/extract_keyframes.py --help >/tmp/drivelm_extract_keyframes_help.txt
python vqa_dataset/carla_vqa_generator_main.py --help >/tmp/drivelm_carla_vqa_help.txt

echo "script_help_ok"
echo "help files:"
echo "  /tmp/drivelm_extract_keyframes_help.txt"
echo "  /tmp/drivelm_carla_vqa_help.txt"
