"""
<주력 모델>
ResNest + fpn-carafe + scnet모델 조합
"""

_base_ = '../scnet/scnet_ResNest.py'

# norm_cfg = dict(type='BN', requires_grad=True)
# act_cfg=dict(type='ReLU', inplace=True)

model = dict(
    neck=dict(
        type='FPN_CARAFE',
        in_channels=[256, 512, 1024, 2048],
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
            compressed_channels=64)))
