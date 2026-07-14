#!/bin/bash
# Train VoteNet 4-class on SUN RGB-D (bed, table, sofa, chair)
# 
# Prerequisites:
#   1. mmdetection3d installed (pip install mmdet3d)
#   2. SUN RGB-D data prepared in data/sunrgbd/
#   3. Pure PyTorch patches loaded (mmcv_pure_patch.py)
#
# Quick start:
#   bash scripts/train.sh

set -e

CONFIG="${1:-configs/votenet/votenet_8xb16_sunrgbd-3d-4class.py}"
WORK_DIR="${2:-work_dirs/votenet_8xb16_sunrgbd-3d-4class}"
GPUS="${3:-1}"

echo "========================================"
echo "VoteNet 4-class Training"
echo "Config: $CONFIG"
echo "Work Dir: $WORK_DIR"
echo "GPUs: $GPUS"
echo "========================================"

mkdir -p "$WORK_DIR"

PYTHONPATH="$PWD:$PYTHONPATH" python -u train.py "$CONFIG" --work-dir "$WORK_DIR"
