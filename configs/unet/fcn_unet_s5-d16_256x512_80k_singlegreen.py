_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/singlegreen.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

model = dict(
    decode_head=dict(
        loss_decode=dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0,
            class_weight = [1,1],#change class weight
        ),
        num_classes=2,
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000) #只有置信分数在0.7以下的像素值点会被拿来训练。在训练时我们至少要保留100000个像素值点
    ),
    auxiliary_head=dict(
        num_classes =2,
        loss_decode=[dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.4)] #change loss weight 2022/1/12
        )
    )

# resume_from = './work_dirs_new/fcn_unet_s5-d16_256x512_80k/iter_72000.pth'
# evaluation = dict(metric='mDice')
work_dir = './work_dirs_new/fcn_unet_s5-d16_256x512_80k_bright'