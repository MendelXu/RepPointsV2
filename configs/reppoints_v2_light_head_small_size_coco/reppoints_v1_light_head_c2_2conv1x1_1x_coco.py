_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    type='RepPointsLightDetector',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=4,
        norm_cfg=norm_cfg),
    bbox_head=dict(
        type='RepPointsNShareHead',
        num_classes=80,
        in_channels=[256, 256],
        feat_channels=[128, 256],
        point_feat_channels=256,
        aux_point_feat_enhance=False,
        stacked_convs=3,
        num_points=9,
        gradient_mul=0.1,
        head_num=2,
        head_pairs=[0, 1, 1, 1, 1],
        head_type=['light', 'normal'],
        point_strides=[4, 8, 16, 32, 64],
        upsample_cfg=dict(mode='bilinear', align_corners=True),
        norm_cfg=norm_cfg,
        head_fuse_config=dict(
            type='Top2DownFuseLayer',
            add_input_conv=False,
            add_out_conv=False,
            style='fpn',
            num_fuse=1,
            conv_cfg=None,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU')),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_init=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.5),
        loss_bbox_refine=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)
        )
)
# training and testing settings
train_cfg = dict(
    init=dict(
        assigner=dict(type='PointAssignerV2', scale=4, pos_num=1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    refine=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.6),
    max_per_img=100)
optimizer = dict(lr=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2), _delete_=True)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[
                    58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(672, 400), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='LoadRPDV2Annotations'),
    dict(type='RPDV2FormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes',
                               'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(672, 400),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(train=dict(pipeline=train_pipeline), val=dict(
    pipeline=test_pipeline), test=dict(pipeline=test_pipeline))
data_root = 'data/coco/'
# checkpoint
checkpoint_config = dict(create_symlink=False)
