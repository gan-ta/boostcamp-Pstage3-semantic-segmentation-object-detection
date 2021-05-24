"""
swin detection model + Augmentation version2 적용 및 튜닝을 한 dataset
"""

import os

#Configurations
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
        limit=2,
        border_mode=1,
        p=0.8
    ),
    # dict(
    #     type='ShiftScaleRotate',
    #     shift_limit=0.01,
    #     # shift_limit=1/32,    # 16 of 512 px
    #     # shift_limit=1/16,    # 32 of 512 px
    #     scale_limit=0.01,
    #     rotate_limit=20,
    #     border_mode=1,
    #     p=0.8
    # ),
    dict(
        type='RandomSizedBBoxSafeCrop',
        height=512,
        width=512,
        erosion_rate=0.2,
        p=0.8
    ),
    # dict(
    #     type='RandomBrightnessContrast',
    #     brightness_limit=0.05,
    #     contrast_limit=0.05,
    #     p=0.2
    # ),
    # 예시 config에 있지만 안 넣은 augmentations
    # 1) One of [RGBShift, HueSaturationValue]
    # 2) ChannelShuffle
    # 3) One of [Blur, MedianBlur]
    # dict(
    #     type='Cutout',
    #     num_holes=64,
    #     max_h_size=2,
    #     max_w_size=2,
    #     p=0.8
    # ),
    # dict(
    #     type='ElasticTransform',
    #     alpha=10,
    #     sigma=3,
    #     alpha_affine=0,
    #     p=0.8
    # ),
    dict(
        type='CLAHE',
        # clip_limit=(1.5, 4.0),
        # p=0.8,
        clip_limit=(1.1, 3.5),
        p=0.95
    ),
]

albu_test_transforms = [
    dict(
        type='CLAHE',
        # clip_limit=(1.5, 4.0),
        # p=1.0,
        clip_limit=(1.1, 3.5),
        p=0.95,
        always_apply=True
    ),
]

""" Pipelines """
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_label=True, with_mask=True),
    # dict(
    #     type='Resize',
    #     img_scale=(768, 768),
    #     keep_ratio=True
    # ),
    dict(
        type='Resize',
        img_scale=[
             (512, 512), (512, 576), (576, 512),
             (576, 576), (576, 640), (640, 576),
             (640, 640), (640, 704), (704, 640),
             (704, 704), (704, 768), (768, 704),
             (768, 768),
        ],
        multiscale_mode='value',
        keep_ratio=False
    ),
    # dict(
    #     type='Resize',
    #     img_scale=[
    #          (512, 512), (512, 576), (576, 512),
    #          (576, 576), (576, 640), (640, 576),
    #          (640, 640), (640, 704), (704, 640),
    #          (704, 704), (704, 768), (768, 704),
    #          (768, 768), (768, 832), (832, 768),
    #          (832, 832), (832, 896), (896, 832),
    #          (896, 896), (896, 960), (960, 896),
    #          (960, 960), (960, 1024), (1024, 960),
    #          (1024, 1024)
    #     ],
    #     multiscale_mode='value',
    #     keep_ratio=False
    # ),
    # dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='RandomFlip', flip_ratio=[0.5, 0.1], direction=['horizontal', 'vertical']),
    dict(
        type='InstaBoost',
        action_candidate=('normal', 'horizontal', 'vertical', 'skip'),
        action_prob=(1, 0, 0, 0),
        scale=(0.8, 1.2),
        dx=15,
        dy=15,
        theta=(-2, 2),
        color_prob=0.2,
        hflag=True,
        aug_ratio=0.5
    ),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_area=3**2,
            min_visibility=0.2,
            # A simple workaround to remove masks without boxes
            filter_lost_elements=True
        ),
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
    # dict(type='Pad', size_divisor=32),
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
            # 'gt_bboxes': 'bboxes',
            # 'gt_masks': 'masks'
        },
        # update final shape
        update_pad_shape=False,
        # skip_img_without_anno(bool): Whether to skip the image if no ann left after aug
        skip_img_without_anno=True
    ),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        # img_scale=[(512, 512), (576, 576), (640, 640), (704, 704), (768, 768)],
        flip=True,
        transforms=[
            dict(type='Resize', multiscale_mode='value', keep_ratio=True),
#            dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
             dict(type='RandomFlip', flip_ratio=[0.5, 0.2], direction=['horizontal', 'vertical']),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            # dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    )
]

# Dataset
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
        pipeline=test_pipeline
    )
)
