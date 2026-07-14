_base_ = [
    '../_base_/datasets/sunrgbd-3d.py', '../_base_/models/votenet.py',
    '../_base_/schedules/schedule-3x.py', '../_base_/default_runtime.py'
]

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

class_names = ('bed', 'table', 'sofa', 'chair')
metainfo = dict(classes=class_names)

model = dict(
    bbox_head=dict(
        num_classes=4,
        bbox_coder=dict(
            type='PartialBinBasedBBoxCoder',
            num_sizes=4,
            num_dir_bins=12,
            with_rot=True,
            mean_sizes=[
                [2.114256, 1.620300, 0.927272],
                [0.791118, 1.279516, 0.718182],
                [0.923508, 1.867419, 0.845495],
                [0.591958, 0.552978, 0.827272],
            ]),
    ),
    test_cfg=dict(
        sample_mode='seed',
        score_thr=0.03,
        nms_thr=0.25,
        per_class_proposal=True,
    ),
)

train_dataloader = dict(
    batch_size=4,
    num_workers=0,
    dataset=dict(
        dataset=dict(
            ann_file='sunrgbd_infos_train_4class.pkl',
            filter_empty_gt=True,
            metainfo=metainfo,
        )))
val_dataloader = dict(
    num_workers=0,
    dataset=dict(
        ann_file='sunrgbd_infos_val_4class.pkl',
        filter_empty_gt=True,
        metainfo=metainfo,
    ))
test_dataloader = dict(
    dataset=dict(
        ann_file='sunrgbd_infos_val_4class.pkl',
        filter_empty_gt=True,
        metainfo=metainfo,
    ))

auto_scale_lr = dict(enable=False, base_batch_size=128)

optim_wrapper = dict(
    clip_grad=dict(max_norm=10, norm_type=2, error_if_nonfinite=False))
