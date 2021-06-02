model = dict(
    type='CascadeRCNN',
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
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
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
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
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
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
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
        ],
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=11,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))
dataset_type = 'CocoDataset'
data_root = '../../input/data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
    dict(type='Rotate', limit=8, border_mode=0, p=0.8),
    dict(
        type='RandomSizedBBoxSafeCrop',
        height=512,
        width=512,
        erosion_rate=0.2,
        p=0.8),
    dict(type='CLAHE', clip_limit=(1.5, 4.0), p=0.8)
]
albu_test_transforms = [
    dict(type='CLAHE', clip_limit=(1.5, 4.0), p=0.8, always_apply=True)
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_label=True,
        with_mask=True),
    dict(
        type='Resize',
        img_scale=[(448, 512), (480, 512), (512, 512), (544, 512), (576, 512),
                   (608, 512), (640, 512), (672, 512), (704, 512), (736, 512),
                   (768, 512), (800, 512)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(
        type='RandomFlip',
        flip_ratio=[0.5, 0.1],
        direction=['horizontal', 'vertical']),
    dict(
        type='Albu',
        transforms=[
            dict(type='Rotate', limit=8, border_mode=0, p=0.8),
            dict(
                type='RandomSizedBBoxSafeCrop',
                height=512,
                width=512,
                erosion_rate=0.2,
                p=0.8),
            dict(type='CLAHE', clip_limit=(1.5, 4.0), p=0.8)
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_area=9,
            min_visibility=0.2,
            filter_lost_elements=True),
        keymap=dict(img='image', gt_bboxes='bboxes', gt_masks='masks'),
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor'))
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Albu',
        transforms=[
            dict(
                type='CLAHE', clip_limit=(1.5, 4.0), p=0.8, always_apply=True)
        ],
        bbox_params=dict(type='BboxParams', format='pascal_voc'),
        keymap=dict(img='image'),
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(512, 512)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
classes = ('UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
           'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing')
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        classes=('UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal',
                 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
                 'Clothing'),
        ann_file='../../input/data/train_mosaic.json',
        img_prefix='../../input/data/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadAnnotations',
                with_bbox=True,
                with_label=True,
                with_mask=True),
            dict(
                type='Resize',
                img_scale=[(448, 512), (480, 512), (512, 512), (544, 512),
                           (576, 512), (608, 512), (640, 512), (672, 512),
                           (704, 512), (736, 512), (768, 512), (800, 512)],
                multiscale_mode='value',
                keep_ratio=True),
            dict(
                type='RandomFlip',
                flip_ratio=[0.5, 0.1],
                direction=['horizontal', 'vertical']),
            dict(
                type='Albu',
                transforms=[
                    dict(type='Rotate', limit=8, border_mode=0, p=0.8),
                    dict(
                        type='RandomSizedBBoxSafeCrop',
                        height=512,
                        width=512,
                        erosion_rate=0.2,
                        p=0.8),
                    dict(type='CLAHE', clip_limit=(1.5, 4.0), p=0.8)
                ],
                bbox_params=dict(
                    type='BboxParams',
                    format='pascal_voc',
                    label_fields=['gt_labels'],
                    min_area=9,
                    min_visibility=0.2,
                    filter_lost_elements=True),
                keymap=dict(img='image', gt_bboxes='bboxes', gt_masks='masks'),
                update_pad_shape=False,
                skip_img_without_anno=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
                meta_keys=('filename', 'ori_shape', 'img_shape',
                           'img_norm_cfg', 'pad_shape', 'scale_factor'))
        ]),
    val=dict(
        type='CocoDataset',
        classes=('UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal',
                 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
                 'Clothing'),
        ann_file='../../input/data/val.json',
        img_prefix='../../input/data/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='Albu',
                transforms=[
                    dict(
                        type='CLAHE',
                        clip_limit=(1.5, 4.0),
                        p=0.8,
                        always_apply=True)
                ],
                bbox_params=dict(type='BboxParams', format='pascal_voc'),
                keymap=dict(img='image'),
                update_pad_shape=False,
                skip_img_without_anno=True),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(512, 512)],
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='RandomFlip',
                        flip_ratio=0.5,
                        direction='horizontal'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        classes=('UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal',
                 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
                 'Clothing'),
        ann_file='../../input/data/test.json',
        img_prefix='../../input/data/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='Albu',
                transforms=[
                    dict(
                        type='CLAHE',
                        clip_limit=(1.5, 4.0),
                        p=0.8,
                        always_apply=True)
                ],
                bbox_params=dict(type='BboxParams', format='pascal_voc'),
                keymap=dict(img='image'),
                update_pad_shape=False,
                skip_img_without_anno=True),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(512, 512)],
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='RandomFlip',
                        flip_ratio=0.5,
                        direction='horizontal'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=1, metric=['bbox', 'segm'], save_best='bbox_mAP_50')
epochs = 36
warmup_epochs = 2
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.05)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2,
    warmup_ratio=0.001,
    by_epoch=True,
    step=[4, 12, 14, 20, 22, 30],
    gamma=0.5)
runner = dict(type='EpochBasedRunner', max_epochs=36)
checkpoint_config = dict(max_keep_ckpts=10, interval=1)
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='WandbLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/content/drive/My Drive/Colab Notebooks/swinproject/code/Swin-Transformer-Object-Detection/cascade_mask_rcnn_swin_tiny_patch4_window7.pth'
resume_from = None
workflow = [('train', 1)]
work_dir = './work_dirs/swinB_myconfig'
gpu_ids = range(0, 1)
