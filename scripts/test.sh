#!/bin/bash
# Evaluate VoteNet 4-class model on SUN RGB-D validation set
#
# Usage:
#   bash scripts/test.sh [config] [checkpoint]

CONFIG="${1:-configs/votenet/votenet_8xb16_sunrgbd-3d-4class.py}"
CHECKPOINT="${2:-checkpoints/epoch_22.pth}"

echo "========================================"
echo "VoteNet 4-class Evaluation"
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "========================================"

PYTHONPATH="$PWD:$PYTHONPATH" python -u test.py "$CONFIG" "$CHECKPOINT"
