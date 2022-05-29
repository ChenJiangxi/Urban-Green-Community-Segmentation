_base_ = [
    '../_base_/models/ocrnet_hr18.py', '../_base_/datasets/singlegreen.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
norm_cfg = dict(type='BN', requires_grad=True)

# evaluation = dict(metric='mDice')
work_dir = './work_dirs_new/ocrnet_hr18_256x512_80k'