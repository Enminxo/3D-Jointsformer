# model settings

num_classes = 12
clip_len = 32
model = dict(
    type='SkeletonGCN',
    backbone=dict(
        type='HandsFormer',
        dim=192,
        # channels=708,
        ksize=17,  # for channel wise convolution
        in_joints=21 * 3,
        out_joints=192,
    ),
    cls_head=dict(
        type='CNNHead',
        num_classes=num_classes,
        in_channels=250,  # backbone's output -> 128 channesl, in cls_head -> fc
        loss_cls=dict(type='CrossEntropyLoss')),
    train_cfg=None,
    test_cfg=None)
