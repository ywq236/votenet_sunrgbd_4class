_base_ = [os.path.join(os.path.dirname(__file__), 'votenet_8xb16_sunrgbd-3d-4class.py')]

load_from = 'work_dirs/votenet_8xb16_sunrgbd-3d-4class_ft/epoch_11.pth'
work_dir = './work_dirs/votenet_8xb16_sunrgbd-3d-4class_ft3'

train_cfg = dict(max_epochs=24)

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=24,
        eta_min=1e-5,
        begin=0,
        end=24,
        by_epoch=True,
    )
]

optim_wrapper = dict(
    optimizer=dict(lr=1e-4),
    clip_grad=dict(max_norm=10, norm_type=2, error_if_nonfinite=False))
