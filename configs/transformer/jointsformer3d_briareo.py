# dataset settings
dataset_type = 'PoseDataset'
# ann_file_train = 'data/briareo/RGB/train.pkl'
# ann_file_val = 'data/briareo/RGB/val.pkl'
# ann_file_test = 'data/briareo/RGB/test.pkl'

ann_file_test = 'data/0_288.pkl'
ann_file_val = 'data/0_346.pkl'
ann_file_train = 'data/0_806.pkl'

# path to save the training logs
work_dir = './work_dirs/briareo/RGB/revision/fold0'

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
    videos_per_gpu=32,
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

# model params config
num_classes = 12
max_len = 32
embed_dim = 512
model = dict(
    type='SkeletonGCN',
    backbone=dict(
        type='Jointsformer3D',
        dim=embed_dim,
        heads=8,
        in_joints=21 * 3,
        out_joints=embed_dim,
    ),
    cls_head=dict(
        type='CNNHead',
        num_classes=num_classes,
        in_channels=embed_dim*2,  # 512,1024
        loss_cls=dict(type='CrossEntropyLoss')),
    # train_cfg=None,
    # test_cfg=None
)


# todo: symlink a media/Data/enz/
# optimizer
optimizer = dict(type='Adam', lr=0.001, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0003)  # this lr is used for 8 gpus
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0003)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# learning policy
# lr_config = dict(policy='step', step=[350, 430, 470])
# total_epochs = 500
lr_config = dict(policy='CosineAnnealing', by_epoch=False, min_lr=0)
total_epochs = 100
# runtime settings
checkpoint_config = dict(interval=5)  # save each 5 epochs
evaluation = dict(interval=5, metrics=['top_k_accuracy'])
# evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))

# config to register logger hook
# interval to print the log
# log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])  # 500 epochs
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
# log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])

# hooks= [dict(type='TensorboardLoggerHook') ]

dist_params = dict(backend='nccl')  # Parameters to set up distributed training
log_level = 'INFO'  # The output level of the log.
load_from = None
# Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved.
resume_from = None
# Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once.
# workflow = [('train', 1)] # todo: not tested yet
workflow = [('train', 1), ('val', 1)]
