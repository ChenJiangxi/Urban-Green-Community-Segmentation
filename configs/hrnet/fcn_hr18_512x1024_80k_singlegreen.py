_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/singlegreen.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

norm_cfg = dict(type='BN', requires_grad=True)
crop_size = (256,512)
model = dict(
    decode_head=dict(
        loss_decode=[
            # dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, class_weight = [1,1]),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)
            ],
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)
    ),
    # auxiliary_head=[
    #     dict(
    #         type='FCNHead',
    #         in_channels=[36, 72, 144],
    #         channels=sum([36, 72, 144]),
    #         input_transform='resize_concat',
    #         in_index=(1, 2, 3),
    #         kernel_size=1,
    #         num_convs=1,
    #         norm_cfg=norm_cfg,
    #         concat_input=False,
    #         dropout_ratio=-1,
    #         num_classes=2,
    #         align_corners=False,
    #         loss_decode=dict(
    #             type='DiceLoss', loss_name='loss_dice', use_sigmoid=False, loss_weight=0.4),
    #         ),
    #     dict(
    #         type='FCNHead',
    #         in_channels=[72, 144],
    #         channels=sum([72, 144]),
    #         input_transform='resize_concat',
    #         in_index=(2, 3),
    #         kernel_size=1,
    #         num_convs=1,
    #         norm_cfg=norm_cfg,
    #         concat_input=False,
    #         dropout_ratio=-1,
    #         num_classes=2,
    #         align_corners=False,
    #         loss_decode=dict(
    #             type='DiceLoss', loss_name='loss_dice', use_sigmoid=False, loss_weight=0.4),
    #         ),
    # ]
)

evaluation = dict(metric='mDice')
work_dir = './work_dirs_new/fcn_hr18_256x512_80k'
