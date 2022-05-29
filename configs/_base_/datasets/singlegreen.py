# dataset settings
dataset_type = 'SingleGreenDataset'
data_root = 'data/singlegreen_new'
img_norm_cfg = dict(
    mean=[62.14484866,67.69474223,65.19883835], std=[39.26080611,38.97791043,40.7339263 ], to_rgb=True)
    # mean=[80.0019443,86.98287457,83.51913964], std=[50.61495346,49.13560703,51.61253566 ], to_rgb=True)
    # mean=[123.675, 116.28, 103.53],
    # std=[58.395, 57.12, 57.375],
    # to_rgb=True)
crop_size = (256,512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='RandomMosaic', prob=0.3, img_scale=(512,1024)),
    # dict(type='Resize', img_scale=(512, 1024), keep_ratio=True),
    dict(type='Resize', img_scale=(512, 1024), multiscale_mode='range', ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 1024),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2, #batch_size
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        # img_dir=['rawdata/train','rawdata/pesudo'],
        # ann_dir=['labeldata/train', 'labeldata/pesudo_hrnet'],
        # split = ['splits/train.txt', 'splits/pesudo_100.txt'],
        img_dir='rawdata/train',
        ann_dir='labeldata/train',
        split = 'splits/train.txt', 
        pipeline=train_pipeline 
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='rawdata/val',
        ann_dir='labeldata/val',
        split="splits/val.txt",
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='rawdata/test',
        ann_dir='labeldata/test_new',
        split="splits/test.txt",
        pipeline=test_pipeline))

