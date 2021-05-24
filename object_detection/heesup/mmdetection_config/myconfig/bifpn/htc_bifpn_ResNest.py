"""
ResNest + BiFPN(neck 튜닝) + HTC 모델조합
"""
_base_ = '../htc/htc_ResNest.py'

model = dict(
    neck=dict(
        # model의 neck 부분만 BiFPN으로 변경
        type='BiFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        strides=[4, 8, 16, 32],
        norm_cfg=dict(type='BN', requires_grad=True),
        num_outs=5)
)
