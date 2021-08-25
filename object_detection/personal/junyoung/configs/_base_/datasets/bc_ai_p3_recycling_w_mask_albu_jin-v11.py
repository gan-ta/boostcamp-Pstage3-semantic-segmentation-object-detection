import os


""" Configurations """
# Paths
data_dir = r'/opt/ml/input/data/'
# Dataset
dataset_type = 'CocoDataset'
classes = (
    "UNKNOWN", "General trash", "Paper", "Paper pack", "Metal",
    "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery",
    "Clothing"
)
# Normalization
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

""" Augmentation """
albu_train_transforms = [
    dict(
        type='Rotate',
        limit=10,
        border_mode=0,
        p=0.8
    ),
    dict(
        type='RandomSizedBBoxSafeCrop',
        height=512,
        width=512,
        erosion_rate=0.2,
        p=0.8
    ),
    dict(
        type='CLAHE',
        clip_limit=(1.5, 4.0),
        p=0.8
    ),
]
albu_test_transforms = [
    dict(
        type='CLAHE',
        clip_limit=(1.5, 4.0),
        p=1.0,
        always_apply=True
    ),
]

""" Pipelines """
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_label=True, with_mask=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=[0.5, 0.1], direction=['horizontal', 'vertical']),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_area=4**2,
            min_visibility=0.2,
            # A simple workaround to remove masks without boxes
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes',
            'gt_masks': 'masks'
        },
        # update final shape
        update_pad_shape=False,
        # skip_img_without_anno(bool): Whether to skip the image if no ann left after aug
        skip_img_without_anno=True
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg', 'pad_shape', 'scale_factor')
    )
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Albu',
        transforms=albu_test_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc'
        ),
        keymap={
            'img': 'image',
        },
        # update final shape
        update_pad_shape=False,
        # skip_img_without_anno(bool): Whether to skip the image if no ann left after aug
        skip_img_without_anno=True
    ),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(512, 512)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ]
    )
]

""" Dataset """
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=os.path.join(data_dir, r'train.json'),
        img_prefix=data_dir,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=os.path.join(data_dir, r'val.json'),
        img_prefix=data_dir,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=os.path.join(data_dir, r'test.json'),
        img_prefix=data_dir,
        pipeline=test_pipeline))
