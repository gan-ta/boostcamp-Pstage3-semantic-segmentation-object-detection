epochs = 12 * 3
warmup_epochs = 2
""" Sweep """
# epochs = 4
# warmup_epochs = 1

""" Optimizer """
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.05)
optimizer_config = dict(grad_clip=None)

""" Scheduler """
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_by_epoch=True,
#     warmup_iters=warmup_epochs,
#     # warmup_ratio=1e-1,
#     # warmup_ratio=1e-2,
#     warmup_ratio=1e-3,
#     by_epoch=True,
#     # SCNet, HTC
#     # step=[6 - warmup_epochs, 12, 20, 22, 30],
#     step=[6 - warmup_epochs, 12, 14, 20, 22, 30],
#     # step=[6 - warmup_epochs, 10, 16, 24],
#     # step=list(range(6 - warmup_epochs, epochs, 2)),
#     # Faster R-CNN
#     # step=list(range(6 - warmup_epochs, epochs, 2)),
#     gamma=0.5
# )

# """ Runner """
# runner = dict(type='EpochBasedRunnerAmp', max_epochs=epochs)


lr_config = dict(policy='CosineAnnealing',warmup='linear',warmup_iters=3000,
                    warmup_ratio=0.0001, min_lr_ratio=1e-7)
runner = dict(type='EpochBasedRunnerAmp', max_epochs=60)