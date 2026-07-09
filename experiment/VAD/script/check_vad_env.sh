#!/bin/bash

echo "========== VAD Environment Check =========="

echo "Python:"
python --version

echo ""

echo "PyTorch:"
python - <<EOF
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF


echo ""

echo "MMCV:"
python - <<EOF
import mmcv
print("mmcv:", mmcv.__version__)
EOF


echo ""

echo "MMDetection:"
python - <<EOF
import mmdet
print("mmdet:", mmdet.__version__)
EOF


echo ""

echo "MMDetection3D:"
python - <<EOF
import mmdet3d
print("mmdet3d:", mmdet3d.__version__)
EOF


echo ""

echo "========== Done =========="