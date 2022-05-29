_base_ = [
    '../_base_/models/fcn_r50-d8.py', '../_base_/datasets/CommunityGreen.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_320k.py'
]

model = dict(
    decode_head=dict(
        num_classes=4,
        loss_decode = dict(
            class_weight = [0.1,0.1,0.1,1]
        )
    ),
    auxiliary_head=dict(num_classes =4))
work_dir = './work_dirs/fcn_r50_256x512_320k_4label'