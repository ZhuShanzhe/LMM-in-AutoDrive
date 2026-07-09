#!/bin/bash

set -e


ROOT=$(pwd)

echo "========== Prepare NuScenes Data =========="


python tools/data_converter/vad_nuscenes_converter.py \
    nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag vad_nuscenes \
    --version v1.0-mini \
    --canbus ./data


echo ""

echo "Generated files:"

ls -lh data/nuscenes/*.pkl


echo "========== Done =========="