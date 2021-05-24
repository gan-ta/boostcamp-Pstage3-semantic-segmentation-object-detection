"""
mmdetection 사용 시 runtime시 사용한 설정
"""
checkpoint_config = dict(interval=1)

# yapf:disable
# wandb 설정
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='mmdetection_trash')
        )
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

# fp16 = dict(loss_scale=512.) # Mixed precision

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

evaluation = dict(metric=['bbox', 'segm'],save_best="bbox_mAP_50")
# evaluation = dict(metric=['bbox'])
checkpoint_config = dict(max_keep_ckpts=3, interval=1)
