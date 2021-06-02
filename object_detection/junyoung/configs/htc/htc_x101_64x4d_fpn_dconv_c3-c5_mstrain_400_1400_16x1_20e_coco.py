_base_ = './htc_r50_fpn_1x_coco.py'
model = dict(
    pretrained=True,
    backbone=dict(
        type='EfficientNet',
        model_type='efficientnet-b0',  # Possible types: ['efficientnet-b0' ... 'efficientnet-b7']
        out_indices=(0, 1, 3, 6)))  # Possible indices: [0 1 2 3 4 5 6],)
# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    dict(
        type='Resize',
        img_scale=[(1600, 400), (1600, 1400)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='SegRescale', scale_factor=1 / 8),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]
data = dict(
    samples_per_gpu=1, workers_per_gpu=1, train=dict(pipeline=train_pipeline))
# learning policy
lr_config = dict(step=[16, 19])
runner = dict(type='EpochBasedRunner', max_epochs=20)
