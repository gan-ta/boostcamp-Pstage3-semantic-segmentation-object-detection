epochs = 12 * 3
warmup_epochs = 1

""" Optimizer """
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=1e-4)
# optimizer_config = dict(grad_clip=None)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

""" Scheduler """
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=warmup_epochs,
    warmup_ratio=1e-3,
    by_epoch=True,
    step=list(range(6 - warmup_epochs, epochs, 4)),
    gamma=0.8
)

""" Runner """
runner = dict(type='EpochBasedRunner', max_epochs=epochs)
