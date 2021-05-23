"""
HRNet + HRFPN + fasterRCNN 조합의 모델
"""

_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]


model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w40',
    backbone=dict(
        _delete_=True,
        type='HRNet',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(40, 80)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(40, 80, 160)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(40, 80, 160, 320)))),
    neck=dict(
        _delete_=True,
        type='HRFPN',
        in_channels=[40, 80, 160, 320],
        out_channels=256))

# model = dict(
#     pretrained='open-mmlab://msra/hrnetv2_w40',
#     backbone=dict(
#         type='HRNet',
#         extra=dict(
#             stage2=dict(num_channels=(40, 80)),
#             stage3=dict(num_channels=(40, 80, 160)),
#             stage4=dict(num_channels=(40, 80, 160, 320)))),
#     neck=dict(type='HRFPN', in_channels=[40, 80, 160, 320], out_channels=256))

# learning policy
lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
