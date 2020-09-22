from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init

from mmdet.core import (PointGenerator, build_assigner, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from mmdet.ops import BRPool, DeformConv, TLPool
from ..builder import HEADS, build_loss, build_fuse_layer
from .reppoints_v2_head import RepPointsV2Head


@HEADS.register_module()
class RepPointsV2NShareHead(RepPointsV2Head):
    """RepPoint head.

    Args:
        head_num (int): number of detector head
        head_type (list): head_type for each head. Lenght should be `head_num`. Choice: normal or light
        head_fuse (bool): whether to fuse features after feature embedding.
        head_fuse_config (dict): config for fuse layer. If `head_fuse` is False, this term whill make no sense. 
        aux_point_feat_enhance (bool): Whether to use thicker head. If it is `True`, we will add another conv module after deformable convolution.
        point_feat_channels (int): Number of channels of points features.
        gradient_mul (float): The multiplier to gradients from
            points refinement and recognition.
        point_strides (Iterable): points strides.
        point_base_scale (int): bbox scale for assigning labels.
        loss_cls (dict): Config of classification loss.
        loss_bbox_init (dict): Config of initial points loss.
        loss_bbox_refine (dict): Config of points loss in refinement.
        use_grid_points (bool): If we use bounding box representation, the
        reppoints is represented as grid points on the bounding box.
        center_init (bool): Whether to use center point assignment.
        transform_method (str): The methods to transform RepPoints to bbox.
    """  # noqa: W605

    def __init__(self,
                 head_num=1,
                 head_type=['normal'],
                 head_pairs=[0, 0, 0, 0, 0],
                 head_fuse=True,
                 head_fuse_config=dict(type='Top2DownFuseLayer'),
                 aux_point_feat_enhance=False,
                 upsample_cfg=dict(mode='nearest'), **kwargs):
        self.head_num = head_num
        self.head_pairs = head_pairs
        self.upsample_cfg = upsample_cfg
        self.head_fuse = head_fuse
        self.head_fuse_config = head_fuse_config
        self.head_type = head_type
        self.aux_point_feat_enhance = aux_point_feat_enhance
        super().__init__(**kwargs)
        assert max(head_pairs) == self.head_num-1 and min(head_pairs) == 0
        

        print(self)
        
    def _init_layers(self):
        """Initialize layers of the head."""
        assert isinstance(self.feat_channels, list) and len(
            self.feat_channels) == self.head_num
        assert isinstance(self.in_channels, list) and len(
            self.in_channels) == self.head_num
        assert isinstance(self.head_type, list)

        if len(self.head_type) == 1:
            self.head_type = self.head_type*self.head_num
        assert len(self.head_type) == self.head_num
        branches = 'cls_convs, reg_convs, shared_convs, hem_tl, hem_br, reppoints_cls_conv, reppoints_cls_out, reppoints_pts_init_conv, reppoints_pts_init_out,reppoints_pts_refine_conv, reppoints_pts_refine_out, reppoints_hem_tl_score_out, reppoints_hem_br_score_out, reppoints_hem_tl_offset_out, reppoints_hem_br_offset_out, reppoints_sem_out, reppoints_sem_embedding'.split(
            ',')
        for branch in branches:
            setattr(self, branch.strip(), nn.ModuleList())

        for i in range(self.head_num):
            subnets = self._init_head(
                self.in_channels[i], self.feat_channels[i], self.head_type[i])
            assert len(subnets) == len(branches)
            for branch, subnet in zip(branches, subnets):
                getattr(self, branch.strip()).append(subnet)
        # fuse feature from second branch to the first branch
        if self.head_fuse:
            in_channels = [self.feat_channels[idx] for idx in self.head_pairs]
            out_channels = in_channels
            self.head_fuse_config.update(dict(in_channels=in_channels,out_channels=out_channels))
            self.fuse_loc = build_fuse_layer(self.head_fuse_config)
            self.fuse_cls = build_fuse_layer(self.head_fuse_config)

    def _init_head(self, in_channels, feat_channels, head_type):

        self.relu = nn.ReLU(inplace=True)
        # feat
        cls_convs, reg_convs, shared_convs = self._init_feature_branch(
            in_channels, feat_channels, head_type)
        # aux
        hem_tl = TLPool(feat_channels, self.conv_cfg, self.norm_cfg,
                        first_kernel_size=self.first_kernel_size, kernel_size=self.kernel_size, corner_dim=self.corner_dim)
        hem_br = BRPool(feat_channels, self.conv_cfg, self.norm_cfg,
                        first_kernel_size=self.first_kernel_size, kernel_size=self.kernel_size, corner_dim=self.corner_dim)
        reppoints_hem_tl_score_out = nn.Conv2d(
            feat_channels, 1, 3, 1, 1)
        reppoints_hem_br_score_out = nn.Conv2d(
            feat_channels, 1, 3, 1, 1)
        reppoints_hem_tl_offset_out = nn.Conv2d(
            feat_channels, 2, 3, 1, 1)
        reppoints_hem_br_offset_out = nn.Conv2d(
            feat_channels, 2, 3, 1, 1)

        reppoints_sem_out = nn.Conv2d(
            feat_channels, self.cls_out_channels, 1, 1, 0)
        reppoints_sem_embedding = ConvModule(
            feat_channels,
            feat_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)
        # predictor head
        pts_out_dim = 4 if self.use_grid_points else 2 * self.num_points
        cls_in_channels = feat_channels + 6
        # first stage
        # classification
        if self.aux_point_feat_enhance:
            reppoints_cls_conv = DeformConv(cls_in_channels,
                                            feat_channels,
                                            self.dcn_kernel, 1, self.dcn_pad)
            reppoints_cls_out = nn.Sequential(
                ConvModule(feat_channels, self.point_feat_channels,
                           1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, order=('norm','act','conv')),
                nn.ReLU(True),
                nn.Conv2d(self.point_feat_channels,
                          self.cls_out_channels, 1, 1, 0)
            )
        else:
            reppoints_cls_conv = DeformConv(cls_in_channels,
                                            self.point_feat_channels,
                                            self.dcn_kernel, 1, self.dcn_pad)
            reppoints_cls_out = nn.Sequential(
                                    nn.ReLU(True),
                                    nn.Conv2d(self.point_feat_channels,
                                          self.cls_out_channels, 1, 1, 0)
                                    )
        # regression
        if head_type in ['light', 'light_head']:
            reppoints_pts_init_conv = nn.Conv2d(feat_channels,
                                                self.point_feat_channels, 1, stride=1, padding=0)
        else:
            reppoints_pts_init_conv = nn.Conv2d(feat_channels,
                                                self.point_feat_channels, 3, stride=1, padding=1)
        reppoints_pts_init_out = nn.Conv2d(self.point_feat_channels,
                                           pts_out_dim, 1, 1, 0)
        # second stage
        # regression
        pts_in_channels = feat_channels + 6
        if self.aux_point_feat_enhance:
            reppoints_pts_refine_conv = DeformConv(pts_in_channels,
                                                   feat_channels,
                                                   self.dcn_kernel, 1,
                                                   self.dcn_pad)

            reppoints_pts_refine_out = nn.Sequential(
                ConvModule(feat_channels, self.point_feat_channels,
                           1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg,order=('norm','act','conv')),
                nn.ReLU(True),
                nn.Conv2d(self.point_feat_channels,
                          pts_out_dim, 1, 1, 0)
            )
        else:
            reppoints_pts_refine_conv = DeformConv(pts_in_channels,
                                                   self.point_feat_channels,
                                                   self.dcn_kernel, 1,
                                                   self.dcn_pad)

            reppoints_pts_refine_out = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(self.point_feat_channels,
                                                 pts_out_dim, 1, 1, 0)
                )

        return cls_convs, reg_convs, shared_convs, hem_tl, hem_br, reppoints_cls_conv, reppoints_cls_out, reppoints_pts_init_conv, reppoints_pts_init_out, reppoints_pts_refine_conv, reppoints_pts_refine_out, reppoints_hem_tl_score_out, reppoints_hem_br_score_out, reppoints_hem_tl_offset_out, reppoints_hem_br_offset_out, reppoints_sem_out, reppoints_sem_embedding

    def _init_feature_branch(self, in_channels, feat_channels, head_type):
        cls_convs = nn.ModuleList()
        reg_convs = nn.ModuleList()
        shared_convs = nn.ModuleList()
        if head_type in ['light', 'light_feat']:
            cls_convs.append(
                ConvModule(in_channels, feat_channels, 1, stride=1,
                           padding=0, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)
            )
            reg_convs.append(
                ConvModule(in_channels, feat_channels, 1, stride=1,
                           padding=0, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)
            )
        else:
            for i in range(self.stacked_convs):
                chn = in_channels if i == 0 else feat_channels
                cls_convs.append(
                    ConvModule(
                        chn,
                        feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
                reg_convs.append(
                    ConvModule(
                        chn,
                        feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
        for i in range(self.shared_stacked_convs):
            shared_convs.append(
                ConvModule(
                    feat_channels,
                    feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        return cls_convs, reg_convs, shared_convs

    def init_weights(self):
        """Initialize weights of the head."""
        bias_cls = bias_init_with_prob(0.01)
        def param_init(m,**kwargs):
            if isinstance(m, nn.Conv2d) or isinstance(m, DeformConv):
                normal_init(m, **kwargs)

        zero_init= partial(param_init,std=0.01)
        cls_init = partial(param_init,std=0.01,bias=bias_cls)

        self.cls_convs.apply(zero_init)
        self.reg_convs.apply(zero_init)
        self.shared_convs.apply(zero_init)

        self._init_list(self.reppoints_cls_conv,zero_init)
        self._init_list(self.reppoints_cls_out,cls_init)
        # apply failed
        self._init_list(self.reppoints_pts_init_conv,zero_init)
        self._init_list(self.reppoints_pts_init_out,zero_init)
        self._init_list(self.reppoints_pts_refine_conv,zero_init)
        self._init_list(self.reppoints_pts_refine_out,zero_init)
        #aaply success
        self._init_list(self.reppoints_hem_tl_score_out,cls_init)
        self._init_list(self.reppoints_hem_br_score_out,cls_init)
        self._init_list(self.reppoints_hem_tl_offset_out,zero_init)
        self._init_list(self.reppoints_hem_br_offset_out,zero_init)
        self._init_list(self.reppoints_sem_out,cls_init)

        if self.head_fuse:
            self.fuse_cls.init_weights()
            self.fuse_loc.init_weights()

    def _init_list(self,module,init_func):
        for m in module:
            if isinstance(m,ConvModule):
                init_func(m.conv)
            elif isinstance(m,(nn.Conv2d,DeformConv)):
                init_func(m)
            elif isinstance(m,(nn.Sequential,nn.ModuleList)):
                self._init_list(m,init_func)
    
    def forward(self, feats):
        if not self.head_fuse:
            return multi_apply(self.forward_single, feats, list(range(len(feats))))
        else:
            cls_feats, pts_feats = multi_apply(
                self.forward_feature_branch, feats, list(range(len(feats))))
            cls_feats = self.fuse_cls(cls_feats)
            pts_feats = self.fuse_loc(pts_feats)
            return multi_apply(self.forward_predicition_branch, cls_feats, pts_feats, pts_feats, list(range(len(feats))))

    def forward_single(self, x, idx):
        """ Forward feature map of a single FPN level."""
        cls_feat, pts_feat = self.forward_feature_branch(x, idx)
        shared_feat = pts_feat
        return self.forward_predicition_branch(cls_feat, pts_feat, shared_feat, idx)
    def forward_predicition_branch(self, cls_feat, pts_feat, shared_feat, idx):
        head_index = self.head_pairs[idx]
        dcn_base_offset = self.dcn_base_offset.type_as(cls_feat)
        # If we use center_init, the initial reppoints is from center points.
        # If we use bounding bbox representation, the initial reppoints is
        #   from regular grid placed on a pre-defined bbox.
        if self.use_grid_points or not self.center_init:
            scale = self.point_base_scale / 2
            # scale = self.point_base_scale[idx] / 2
            points_init = dcn_base_offset / dcn_base_offset.max() * scale
            bbox_init = cls_feat.new_tensor([-scale, -scale, scale,
                                             scale]).view(1, 4, 1, 1)
        else:
            points_init = 0
        for shared_conv in self.shared_convs[head_index]:
            shared_feat = shared_conv(shared_feat)

        sem_feat = shared_feat
        hem_feat = shared_feat

        sem_scores_out = self.reppoints_sem_out[head_index](sem_feat)
        sem_feat = self.reppoints_sem_embedding[head_index](sem_feat)

        cls_feat = cls_feat + sem_feat
        pts_feat = pts_feat + sem_feat
        hem_feat = hem_feat + sem_feat
        # generate heatmap and offset
        hem_tl_feat = self.hem_tl[head_index](hem_feat)
        hem_br_feat = self.hem_br[head_index](hem_feat)

        hem_tl_score_out = self.reppoints_hem_tl_score_out[head_index](
            hem_tl_feat)
        hem_tl_offset_out = self.reppoints_hem_tl_offset_out[head_index](
            hem_tl_feat)
        hem_br_score_out = self.reppoints_hem_br_score_out[head_index](
            hem_br_feat)
        hem_br_offset_out = self.reppoints_hem_br_offset_out[head_index](
            hem_br_feat)

        hem_score_out = torch.cat([hem_tl_score_out, hem_br_score_out], dim=1)
        hem_offset_out = torch.cat(
            [hem_tl_offset_out, hem_br_offset_out], dim=1)

        # initialize reppoints
        pts_out_init = self.reppoints_pts_init_out[head_index](
            self.relu(self.reppoints_pts_init_conv[head_index](pts_feat)))
        if self.use_grid_points:
            pts_out_init, bbox_out_init = self.gen_grid_from_reg(
                pts_out_init, bbox_init.detach())
        else:
            pts_out_init = pts_out_init + points_init
        # refine and classify reppoints
        pts_out_init_grad_mul = (
            1 - self.gradient_mul) * pts_out_init.detach() + self.gradient_mul * pts_out_init
        dcn_offset = pts_out_init_grad_mul - dcn_base_offset

        hem_feat = torch.cat([hem_score_out, hem_offset_out], dim=1)
        cls_feat = torch.cat([cls_feat, hem_feat], dim=1)
        pts_feat = torch.cat([pts_feat, hem_feat], dim=1)

        cls_out = self.reppoints_cls_out[head_index](
            self.reppoints_cls_conv[head_index](cls_feat, dcn_offset))
        pts_out_refine = self.reppoints_pts_refine_out[head_index](
            self.reppoints_pts_refine_conv[head_index](pts_feat, dcn_offset))
        if self.use_grid_points:
            pts_out_refine, _ = self.gen_grid_from_reg(
                pts_out_refine, bbox_out_init.detach())
        else:
            pts_out_refine = pts_out_refine + pts_out_init.detach()
        return cls_out, pts_out_init, pts_out_refine, hem_score_out, hem_offset_out, sem_scores_out

    def forward_feature_branch(self, x, idx):
        head_index = self.head_pairs[idx]
        cls_feat = x
        pts_feat = x
        for cls_conv in self.cls_convs[head_index]:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs[head_index]:
            pts_feat = reg_conv(pts_feat)
        return cls_feat, pts_feat

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Hack some keys of the model state dict so that can load checkpoints
        of previous version."""
        local_metadata['version']='2.2'
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)
