epochs = 12 * 3
warmup_epochs = 2
""" Sweep """
# epochs = 4
# warmup_epochs = 1

""" Optimizer """
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.05)
optimizer_config = dict(grad_clip=None)

""" Scheduler """
# 27: 1e-5, 33: 1e-6
# lr_config = dict(step=[27, 33])
# 12: 5e-5, 18: 2.5e-5, 20: 1.25e-5, 28: 6.25e-6, 30: 3.125e-06, 32: 1.5625e-06
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=warmup_epochs,
    # warmup_ratio=1e-1,
    # warmup_ratio=1e-2,
    warmup_ratio=1e-3,
    by_epoch=True,
    # SCNet, HTC
    step=[6 - warmup_epochs, 12, 18, 20, 30],
    # step=[6 - warmup_epochs, 12, 18, 20, 28, 30, 32],
    # step=[6 - warmup_epochs, 12, 14, 20, 22, 30],
    # step=[6 - warmup_epochs, 10, 16, 24],
    # step=list(range(6 - warmup_epochs, epochs, 2)),
    # Faster R-CNN
    # step=list(range(6 - warmup_epochs, epochs, 2)),
    gamma=0.5
)

""" Runner """
runner = dict(type='EpochBasedRunner', max_epochs=epochs)
