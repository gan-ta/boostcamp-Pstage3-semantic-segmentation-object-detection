_base_ = [
    '../_base_/models/cascade_mask_rcnn_swin_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False
    ),
    neck=dict(in_channels=[128, 256, 512, 1024]),
    roi_head=dict(
        bbox_head=[
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=11,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=11,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=11,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
        ]))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

""" Augmentation """
albu_train_transforms = [
    dict(
        type='Rotate',
        limit=8,
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
        # clip_limit=(1.1, 3.0),
        # p=0.95,
        always_apply=True
    ),
]

""" Pipelines """
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_label=True, with_mask=True),
    dict(type='Resize',
         img_scale=[(448, 512), (480, 512), (512, 512), (544, 512), (576, 512),
                     (608, 512), (640, 512), (672, 512), (704, 512),
                     (736, 512), (768, 512), (800, 512)],
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=[0.5, 0.1], direction=['horizontal', 'vertical']),
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

data = dict(train=dict(pipeline=train_pipeline),
            val=dict(pipeline=test_pipeline),
            test=dict(pipeline=test_pipeline))


optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)

load_from = "/content/drive/MyDrive/object detection/cascade_mask_rcnn_swin_base_patch4_window7.pth"
