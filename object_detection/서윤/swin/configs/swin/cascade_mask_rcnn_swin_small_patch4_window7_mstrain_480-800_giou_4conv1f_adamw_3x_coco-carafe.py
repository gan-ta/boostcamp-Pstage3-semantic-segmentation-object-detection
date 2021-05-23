_base_ = [
    '../_base_/models/cascade_mask_rcnn_swin_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_3x_adamw_step.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        ape=False,
        drop_path_rate=0.2,
        patch_norm=True,
        use_checkpoint=False
    ),
    neck=dict(
        type='FPN_CARAFE',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5,
        start_level=0,
        end_level=-1,
        norm_cfg=None,
        act_cfg=None,
        order=('conv', 'norm', 'act'),
        upsample_cfg=dict(
            type='carafe',
            up_kernel=5,
            up_group=1,
            encoder_kernel=3,
            encoder_dilation=1,
            compressed_channels=64)),
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
    dict(
        type='Resize',
        img_scale=(512, 512),
        keep_ratio=True
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
    #     keep_ratio=True
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
        # img_scale=[(256, 256), (512, 512), (768, 768)],
        # img_scale=[(256, 256), (512, 512), (768, 768), (1024, 1024)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
            # dict(type='RandomFlip', flip_ratio=[0.5, 0.2], direction=['horizontal', 'vertical']),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            # dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    )
]

data = dict(train=dict(pipeline=train_pipeline),
            val=dict(pipeline=test_pipeline),
            test=dict(pipeline=test_pipeline))


log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='mmdetection')
        )
    ])


load_from = "/opt/ml/code/Swin-Transformer-Object-Detection/cascade_mask_rcnn_swin_small_patch4_window7.pth"
#resume_from = "/opt/ml/code/Swin-Transformer-Object-Detection/work_dirs/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x/epoch_14.pth"