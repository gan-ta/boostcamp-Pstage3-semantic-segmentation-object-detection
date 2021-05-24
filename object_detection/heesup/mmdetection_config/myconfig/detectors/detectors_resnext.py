"""
ResNext detectors 모델 조합(RFP)
"""

_base_ = [
    '../_base_/models/faster_rcnn_resnext.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        type='DetectoRS_ResNeXt',
        conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True),
        output_img=True),
    neck=dict(
        type='RFP',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        rfp_steps=2,
        aspp_out_channels=64,
        aspp_dilations=(1, 3, 6, 1),
        rfp_backbone=dict(
            rfp_inplanes=256,
            type='DetectoRS_ResNeXt',
            depth=101,
            groups=32,
            base_width=4,            
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            conv_cfg=dict(type='ConvAWS'),
            sac=dict(type='SAC', use_deform=True),
            stage_with_sac=(False, True, True, True),
            pretrained='open-mmlab://resnext101_32x4d',
            style='pytorch')),
 
#     roi_head=dict(
#         bbox_head=[
#             dict(
#                 type='Shared2FCBBoxHead',
#                 in_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 # trash dataset에 맞게 11 class로 변경
#                 num_classes=11,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.1, 0.1, 0.2, 0.2]),
#                 reg_class_agnostic=True,
#                 loss_cls=dict(
#                     type='CrossEntropyLoss',
#                     use_sigmoid=False,
#                     loss_weight=1.0),
#                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
#                                loss_weight=1.0)),
#             dict(
#                 type='Shared2FCBBoxHead',
#                 in_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 # trash dataset에 맞게 11 class로 변경
#                 num_classes=11,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.05, 0.05, 0.1, 0.1]),
#                 reg_class_agnostic=True,
#                 loss_cls=dict(
#                     type='CrossEntropyLoss',
#                     use_sigmoid=False,
#                     loss_weight=1.0),
#                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
#                                loss_weight=1.0)),
#             dict(
#                 type='Shared2FCBBoxHead',
#                 in_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 # trash dataset에 맞게 11 class로 변경
#                 num_classes=11,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.033, 0.033, 0.067, 0.067]),
#                 reg_class_agnostic=True,
#                 loss_cls=dict(
#                     type='CrossEntropyLoss',
#                     use_sigmoid=False,
#                     loss_weight=1.0),
#                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
#         ]
#     )
)

# 저장할 weight를 3개로 지정
checkpoint_config = dict(max_keep_ckpts=3, interval=1)
