# dataset settings
dataset_type = 'PoseDataset'

ann_file_train = './data/briareo/IR/train.pkl'
ann_file_val = './data/briareo/IR/val.pkl'
ann_file_test = './data/briareo/IR/test.pkl'

train_pipeline = [
    # dict(type='PoseRandomCrop', min_ratio=0.5, max_ratio=1.0, min_len=64),
    dict(type='PoseCenterCrop', clip_ratio=0.9),

    dict(type='PoseResize', clip_len=32),

    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),

    dict(type='ToTensor', keys=['keypoint']),
]
val_pipeline = [
    dict(type='PoseCenterCrop', clip_ratio=0.9),

    dict(type='PoseResize', clip_len=32),

    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),

    dict(type='ToTensor', keys=['keypoint']),

]
test_pipeline = [
    dict(type='PoseCenterCrop', clip_ratio=0.9),

    dict(type='PoseResize', clip_len=32),

    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),

    dict(type='ToTensor', keys=['keypoint']),
]

data = dict(
    videos_per_gpu=64,
    # workers_per_gpu=2,
    workers_per_gpu=0,
    shuffle=True,
    test_dataloader=dict(videos_per_gpu=1),

    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix='',
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix='',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix='',
        pipeline=test_pipeline))
