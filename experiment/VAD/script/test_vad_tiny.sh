#!/bin/bash

set -e


echo "========== VAD Tiny Evaluation =========="


CONFIG=projects/configs/VAD/VAD_tiny_stage_2.py

CHECKPOINT=ckpts/VAD_tiny.pth


python tools/test.py \
    $CONFIG \
    $CHECKPOINT \
    --eval bbox


echo ""

echo "========== Evaluation Finished =========="