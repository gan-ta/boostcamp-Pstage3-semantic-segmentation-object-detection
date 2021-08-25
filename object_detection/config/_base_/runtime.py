""" Baseline code """
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ]
)
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

""" Evaluation """
evaluation = dict(interval=1, metric='bbox', save_best="bbox_mAP_50")
# evaluation = dict(interval=1, metric=['bbox', 'segm'], save_best="bbox_mAP_50")

""" Log """
log_level = 'INFO'
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict()
            # init_kwargs=dict(
            #     project='bc-ai-p3-object-detection',
            # )
        )
    ]
)

""" Checkpoint """
workdir = r'/opt/ml/results/_NEW_EXPERIMENT/'
# checkpoint_config = dict(max_keep_ckpts=10, interval=1)
checkpoint_config = dict(max_keep_ckpts=5, interval=1)

""" Resume """
load_from = None
resume_from = None

""" Others """
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
# workflow = [('train', 1), ('val', 1)]
