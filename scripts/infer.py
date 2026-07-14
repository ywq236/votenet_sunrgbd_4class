#!/usr/bin/env python3
"""VoteNet 4-class 3D object detection inference script.
Detects: bed, table, sofa, chair in indoor point clouds (SUN RGB-D format).

Usage:
    python scripts/infer.py <point_cloud.bin>                    # from .bin file
    python scripts/infer.py <points.npy>                         # from .npy array
    python scripts/infer.py --demo                               # demo on a random val sample
"""

import argparse
import os
import sys
import numpy as np

# Must apply pure PyTorch patches BEFORE any mmcv import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmcv_pure_patch import apply_patches
apply_patches()

from mmengine.config import Config
from mmengine.runner import Runner
from mmdet3d.apis import init_model, inference_detector


def main():
    parser = argparse.ArgumentParser(description='VoteNet 4-class inference')
    parser.add_argument('input', nargs='?', help='Path to point cloud (.bin or .npy)')
    parser.add_argument('--config', default='configs/votenet/votenet_8xb16_sunrgbd-3d-4class.py',
                        help='Model config file')
    parser.add_argument('--checkpoint', default='checkpoints/epoch_22.pth',
                        help='Model checkpoint')
    parser.add_argument('--demo', action='store_true',
                        help='Run demo on a random validation sample')
    parser.add_argument('--output', '-o', help='Output JSON file for results')
    args = parser.parse_args()

    # Load model
    print('Loading model...')
    config = Config.fromfile(args.config)
    model = init_model(config, args.checkpoint, device='cuda:0')

    if args.demo:
        # Run on a random validation sample
        import pickle, random
        data_dir = os.environ.get('SUNRGBD_DATA', '/root/mmdetection3d/data/sunrgbd')
        val_path = os.path.join(data_dir, 'sunrgbd_infos_val_4class.pkl')
        with open(val_path, 'rb') as f:
            val_data = pickle.load(f)
        item = random.choice(val_data['data_list'])
        lidar_rel = item['lidar_points']['lidar_path']
        if not lidar_rel.startswith('points/'):
            lidar_rel = os.path.join('points', lidar_rel)
        lidar_path = os.path.join(data_dir, lidar_rel)
        # Load full 6D points (XYZ + RGB), model will handle with use_dim=[0,1,2]
        raw_points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 6)
        print(f'Demo: loaded {raw_points.shape[0]} points from {lidar_rel}')
    else:
        if not args.input:
            parser.error('Please provide input file or use --demo')
        ext = os.path.splitext(args.input)[1].lower()
        if ext == '.bin':
            raw_points = np.fromfile(args.input, dtype=np.float32).reshape(-1, 6)
        elif ext == '.npy':
            raw_points = np.load(args.input)
        else:
            raise ValueError(f'Unsupported format: {ext}. Use .bin or .npy')
        print(f'Loaded {raw_points.shape[0]} points from {args.input}')

    # Model internally samples 20000 points. Use XYZ only (first 3 columns)
    points = raw_points[:, :3].astype(np.float32)
    print('Running inference...')
    result = inference_detector(model, points)

    # inference_detector returns list of results for each sample
    if isinstance(result, (list, tuple)):
        pred_instances = result[0].pred_instances_3d
    else:
        pred_instances = result.pred_instances_3d

    # Parse results
    boxes_3d = pred_instances.bboxes_3d.tensor.cpu().numpy()
    scores = pred_instances.scores_3d.cpu().numpy()
    labels = pred_instances.labels_3d.cpu().numpy()

    class_names = ['bed', 'table', 'sofa', 'chair']

    print(f'\n{"="*60}')
    print(f'Detected {len(scores)} objects:')
    print(f'{"="*60}')
    print(f'{"Class":<8} {"Score":<8} {"Center (x,y,z)":<30} {"Size (l,w,h)":<30} {"Yaw":<8}')
    print(f'{"-"*60}')

    for i in range(min(len(scores), 50)):
        cls_id = int(labels[i])
        if cls_id >= len(class_names):
            continue
        center = boxes_3d[i, :3]
        size = boxes_3d[i, 3:6]
        yaw = boxes_3d[i, 6]
        print(f'{class_names[cls_id]:<8} {scores[i]:.4f}   '
              f'({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})       '
              f'({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f})     '
              f'{yaw:.3f}')

    # Save JSON if requested
    if args.output:
        import json
        detections = []
        for i in range(len(scores)):
            detections.append({
                'class': class_names[int(labels[i])],
                'score': float(scores[i]),
                'center': boxes_3d[i, :3].tolist(),
                'size': boxes_3d[i, 3:6].tolist(),
                'yaw': float(boxes_3d[i, 6]),
            })
        with open(args.output, 'w') as f:
            json.dump(detections, f, indent=2)
        print(f'\nSaved to {args.output}')


if __name__ == '__main__':
    main()
