"""
shedule_1x 기본 스케줄러 config설정(SGD ->  Adam으로 조정)
"""

# optimizer
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='Adam', lr=5e-5, weight_decay= 1e-6)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
