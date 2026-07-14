# VoteNet 4-Class 3D Object Detection on SUN RGB-D

**Target:** Detect **bed, table, sofa, chair** in indoor point clouds.

**Model:** VoteNet (Deep Hough Voting for 3D Object Detection, ICCV 2019)

---

## Quick Start

### Requirements

```bash
pip install torch==2.1.2 torchvision==0.16.2
pip install mmengine mmcv==2.1.0 mmdet mmdet3d
pip install opencv-python scipy numpy
```

### Inference (one line)

```bash
# Demo on a random validation sample
python scripts/infer.py --demo

# Inference on your own point cloud (.bin or .npy)
python scripts/infer.py /path/to/points.bin

# Save results to JSON
python scripts/infer.py /path/to/points.bin -o results.json
```

**Input:** Point cloud with shape `(N, 6)` — columns `[x, y, z, r, g, b]` in DEPTH coordinate system.

**Output:** 3D bounding boxes with class label, confidence score, center, size, and yaw angle.

### Evaluation

```bash
bash scripts/test.sh
```

## Performance

| Metric | Value |
|--------|-------|
| **mAP@0.25** | 0.60 |
| **mAP@0.50** | 0.24 |

| Class | mAP@0.25 | mAP@0.50 | Status |
|-------|----------|----------|--------|
| bed | 0.92 | **0.59** | Reliable |
| chair | 0.68 | **0.32** | Usable |
| sofa | 0.39 | 0.05 | Limited |
| table | 0.34 | 0.05 | Limited |

Trained on **1,558** SUN RGB-D indoor scenes (filtered from 2,000). Data augmentation: random rotation ±30°, scaling 0.85-1.15, horizontal flip, random point sampling to 20,000 points.

## Training from Scratch

### 1. Prepare Data

Extract SUN RGB-D point clouds from raw data:

```bash
python tools/extract_sunrgbd.py
```

This processes SUN RGB-D raw images into .mat point cloud files, calibration, and labels. Requires the full SUN RGB-D dataset with `SUNRGBDMeta3DBB_v2.mat` annotation file.

### 2. Generate Annotation Files

```bash
cd /path/to/mmdetection3d
PYTHONPATH=. python tools/create_data.py sunrgbd \
    --root-path ./data/sunrgbd --out-dir ./data/sunrgbd \
    --extra-tag sunrgbd --workers 8
```

### 3. Filter to 4 Classes

```python
import pickle
# Filter sunrgbd_infos_train.pkl to only keep bed/table/sofa/chair
# See tools/extract_sunrgbd.py for the filter logic
```

### 4. Start Training

```bash
bash scripts/train.sh
```

### 5. Fine-tune (Optional)

After initial training, fine-tune with low learning rate for better box regression:

```bash
python train.py configs/votenet/votenet_8xb16_sunrgbd-3d-4class_finetune.py
```

## Project Structure

```
├── README.md
├── train.py / test.py          # Modified training/testing entry points (with CUDA patch)
├── mmcv_pure_patch.py          # Pure PyTorch replacements for mmcv CUDA ops
├── configs/votenet/
│   └── votenet_8xb16_sunrgbd-3d-4class.py  # Optimized model config
├── checkpoints/
│   └── epoch_22.pth            # Best trained model weights (~13 MB)
├── scripts/
│   ├── infer.py                # Inference script
│   ├── train.sh                # Training launcher
│   └── test.sh                 # Evaluation launcher
├── tools/
│   └── extract_sunrgbd.py      # Data extraction from raw SUN RGB-D
└── patches/
    └── vote_head.py            # Fixed VoteHead (size_class_targets clamp)
```

## Important Notes

1. **mmcv_pure_patch.py** must be imported BEFORE any mmcv import. It replaces buggy CUDA kernels (FurthestPointSample, BallQuery, GroupPoints, QueryAndGroup, ThreeNN, ThreeInterpolate) with pure PyTorch implementations. This is REQUIRED for inference and training.

2. The model was trained on SUN RGB-D with DEPTH coordinate system. Point clouds from other sources may need coordinate transformation.

3. For best results on bed and chair detection, the model is production-ready. Table and sofa detection has limited accuracy and may benefit from additional data or architectural changes.

## Environment (Training Reference)

| Component | Version |
|-----------|---------|
| Python | 3.10.16 |
| PyTorch | 2.1.2+cu118 |
| CUDA | 11.8 |
| GPU | RTX 4090 (24 GB) |
| MMEngine | 0.10.7 |
| MMCV | 2.1.0 |
| MMDet | 3.3.0 |
| MMDet3D | 1.4.0 |

Training time: ~100 min for 24 epochs on 1,558 scenes (RTX 4090).

## License

This project is based on [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) (Apache 2.0 License).
