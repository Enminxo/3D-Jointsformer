# _base_ = [
#     '../_base_/models/briareo_handsformer.py',
#     '../_base_/datasets/briareo_rgb.py',
#     '../_base_/schedules/adam_500e.py',
#     '../_base_/default_runtime.py']

# dataset settings
dataset_type = 'PoseDataset'

ann_file_train = 'data/briareo/RGB/train.pkl'
ann_file_val = 'data/briareo/RGB/val.pkl'
ann_file_test = 'data/briareo/RGB/test.pkl'

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

# model settings
num_classes = 12
clip_len = 32
embed_dim = 252

model = dict(
    type='SkeletonGCN',
    backbone=dict(
        type='JointsFormer',
        dim=embed_dim,  # hidden dim or embed dim 252//6 = 42; 252//4= 63
        heads=6,
        max_len=clip_len,
        input_dim=3,
        output_dim=embed_dim,  # out_joints = dim
    ),
    cls_head=dict(
        type='CNNHead',
        num_classes=num_classes,
        in_channels=3 * embed_dim,  # backbone's output dimension * 3 =756
        loss_cls=dict(type='CrossEntropyLoss')
    ),
    # train_cfg = dict(
    #     type='EpochBasedTrainLoop', max_epochs=100, val_begin=1, val_interval=1),
    # test_cfg = dict(type='TestLoop')
    train_cfg=None,
    test_cfg=None
)

# optimizer
optimizer = dict(type='Adam', lr=0.001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# Learning rate scheduler config used to register Lr Updater hook
# lr_config = dict(policy='step', step=[350, 430, 470])
lr_config = dict(policy='step', step=[150, 200, 300])
total_epochs = 400

# runtime settings
checkpoint_config = dict(by_epoch=True, interval=5)
evaluation = dict(interval=5, metrics=['top_k_accuracy'], topk=(1, 5))
log_config = dict(interval=50,
                  hooks=[dict(type='TextLoggerHook'),
                         dict(type='TensorboardLoggerHook')])
log_level = 'INFO'

workflow = [('train', 1), ('val', 1)]
# workflow = [('train', 1)]

dist_params = dict(backend='nccl')  # Parameters to set up distributed training

load_from = None
# Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved.
resume_from = None

work_dir = './work_dirs/briareo/RGB/jointsformer0616'
# work_dir = './work_dirs/briareo/RGB/st_jointsformer0616'
