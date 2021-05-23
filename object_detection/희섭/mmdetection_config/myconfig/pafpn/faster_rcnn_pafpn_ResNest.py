"""
ResNest + pafpn + faster RCNN모델 조합
"""
_base_ = '../faster_rcnn/ResNest_faster_rcnn.py'

model = dict(
    neck=dict(
        type='PAFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5))
