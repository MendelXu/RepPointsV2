_base_ = 'reppoints_v2_light_head_c2_2conv1x1_1x_coco.py'
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    neck=dict(
        _delete_=True,
        type='BiFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=False,
        num_outs=4,
        no_norm_on_lateral=False,
        num_repeat=2,
        norm_cfg=norm_cfg
    ),
    bbox_head=dict(
        loss_cls=dict(
            _delete_=True,
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox_init=dict(_delete_=True, type='GIoULoss', loss_weight=1.0),
        loss_bbox_refine=dict(_delete_=True, type='GIoULoss', loss_weight=2.0)
    )
)
