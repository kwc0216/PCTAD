model = dict(
    type="ActionFormer",
    projection=dict(
        type="PCTransformerProj",
        in_channels=256,
        out_channels=64,
        arch=(2, 2, 5),  # layers in embed / stem / branch
        conv_cfg=dict(kernel_size=3, proj_pdrop=0.0),
        norm_cfg=dict(type="LN"),
        attn_cfg=dict(n_head=2, n_mha_win_size=19),
        path_pdrop=0.1,
        use_abs_pe=False,
        max_seq_len=2304,
    ),
    neck=dict(
        type="FPNIdentity",
        in_channels=64,
        out_channels=64,
        num_levels=5,
    ),
    rpn_head=dict(
        type="ActionFormerHead",
        num_classes=7,
        in_channels=64,
        feat_channels=64,
        num_convs=2,
        cls_prior_prob=0.01,
        prior_generator=dict(
            type="PointGenerator",
            strides=[1, 2, 4, 8, 16],
            regression_range=[(0, 4), (4, 8), (8, 16), (16, 32), (32, 10000)],
        ),
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        center_sample="radius",
        center_sample_radius=1.5,
        label_smoothing=0.0,
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
        ),
    ),
)
