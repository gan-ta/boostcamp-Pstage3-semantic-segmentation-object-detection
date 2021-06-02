# optimizer
optimizer = dict(type='Adam', lr=0.0001, weight_decay=0.00001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    step=[8, 16])
runner = dict(type='EpochBasedRunner', max_epochs=25)
