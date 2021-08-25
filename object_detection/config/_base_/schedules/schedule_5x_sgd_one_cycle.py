epochs = 12 * 5
warmup_epochs = 1

""" Optimizer """
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

""" Scheduler """
lr_config = dict(
    policy='OneCycle',
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=warmup_epochs,
    warmup_ratio=1e-3,
    max_lr=0.02,
    total_steps=655 * epochs,
    pct_start=0.0
)

""" Runner """
runner = dict(type='EpochBasedRunner', max_epochs=epochs)
