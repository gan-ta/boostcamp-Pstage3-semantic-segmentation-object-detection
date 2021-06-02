checkpoint_config = dict(interval=1,max_keep_ckpts=10)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
            init_kwargs=dict(
                project='mmdetection',
                name='faster_rcnn_r50_bifpn_1x_coco'))
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]