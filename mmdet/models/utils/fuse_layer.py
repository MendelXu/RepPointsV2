from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from mmcv.cnn import ConvModule, normal_init
from mmdet.models.necks.bifpn import LayerCombineModule

from ..builder import FUSELAYER


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


class Plus(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return torch.stack(inputs, dim=-1).sum(dim=-1)


class NaiveFuseLayer(nn.Module):
    fuse_direction = 'none'
    """FuseLayer

    Args:
        in_channels (list): Input channels for inputs.
        out_channels (list): Output channels for outputs. Note that the channels for input and output are same expect input fusing feature from other input. 
        add_input_conv (bool): Whether to add conv module before fusing. If `False`, conv module only add to fusing source with different channels with fusing target.
        add_out_conv (bool): Whether to add conv module after fusing. If 'False', the fusing result outputs directly.
        style (sty): Choice: fpn or bifpn. Defines how to combine the features from different inputs.
        num_fuse (int): Number of inputs fusing features from other inputs.
    """

    def __init__(self, in_channels,
                 out_channels,
                 add_input_conv=False,
                 add_out_conv=False,
                 style='fpn',
                 num_fuse=1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(
                     mode='bilinear',
                     align_corners=True
                 )):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_input_conv = add_input_conv
        self.add_out_conv = add_out_conv
        self.style = style
        self.num_fuse = num_fuse
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.upsample_cfg = upsample_cfg
        # check params
        assert len(self.in_channels) == len(self.out_channels)
        assert len(self.in_channels) > self.num_fuse

    def init_weights(self):
        def param_init(m, **kwargs):
            if isinstance(m, nn.Conv2d):
                normal_init(m, **kwargs)
        zero_init = partial(param_init, std=0.01)
        self.apply(zero_init)


@FUSELAYER.register_module()
class Top2DownFuseLayer(NaiveFuseLayer):
    fuse_direction = 'top2down'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.fuse_conv = nn.ModuleList()
        self.fuse_combine = nn.ModuleList()
        for i in range(self.num_fuse):
            source_channel = self.out_channels[i+1]
            target_channel = self.in_channels[i]

            if self.add_input_conv or (source_channel != target_channel):
                self.fuse_conv.append(
                    ConvModule(source_channel, target_channel, 3, stride=1,
                               padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
                )
            else:
                self.fuse_conv.append(Identity())
            if self.style == 'bifpn':
                self.fuse_combine.append(LayerCombineModule(2))
            elif self.style == 'fpn':
                self.fuse_combine.append(Plus())

        if self.add_out_conv:
            self.out_conv = nn.ModuleList()
            for i in range(self.num_fuse):
                self.out_conv.append(
                    ConvModule(self.in_channels[i], self.out_channels[i], 3, stride=1,
                               padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))

    def forward(self, feats):
        for i in range(len(self.fuse_conv)-1, -1, -1):
            scaled_feat = F.interpolate(self.fuse_conv[i](
                feats[i+1]), size=feats[i].size()[-2:], **self.upsample_cfg)
            feats[i] = self.fuse_combine[i]([feats[i], scaled_feat])
        if self.add_out_conv:
            for i in range(len(self.out_conv)):
                feats[i] = self.out_conv[i](feats[i])

        return feats


@FUSELAYER.register_module()
class Down2TopFuseLayer(NaiveFuseLayer):
    fuse_direction = 'down2top'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fuse_conv = nn.ModuleList()

        self.fuse_combine = nn.ModuleList()
        for i in range(self.num_fuse):
            source_channel = self.out_channels[i]
            target_channel = self.in_channels[i+1]

            if self.add_input_conv or (source_channel != target_channel):
                self.fuse_conv.append(
                    ConvModule(source_channel, target_channel, 3, stride=1,
                               padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg
                               ))
            else:
                self.fuse_conv.append(Identity())
            if self.style == 'bifpn':
                self.fuse_combine.append(LayerCombineModule(2))
            elif self.style == 'fpn':
                self.fuse_combine.append(Plus())

        if self.add_out_conv:
            self.out_conv = nn.ModuleList()
            for i in range(self.num_fuse):
                self.out_conv.append(
                    ConvModule(self.in_channels[i], self.out_channels[i], 3, stride=1,
                               padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))

    def forward(self, feats):
        for i in range(len(self.fuse_conv)):
            scaled_feat = F.adaptive_max_pool2d(self.fuse_conv[i](
                feats[i]), output_size=feats[i+1].size()[-2:])
            feats[i+1] = self.fuse_combine[i]([feats[i+1], scaled_feat])
        if self.add_out_conv:
            for i in range(len(self.out_conv)):
                feats[i] = self.out_conv[i](feats[i])
        return feats


@FUSELAYER.register_module()
class BiFuseLayer(NaiveFuseLayer):
    fuse_direction = 'bidirectional'

    def __init__(self, num_invert_fuse, invert_style='fpn', invert_act_cfg=None, act_last=False, num_repeat=1, **kwargs):
        super().__init__(**kwargs)
        self.act_last = act_last
        self.top2down = nn.ModuleList([Top2DownFuseLayer(**kwargs)]*num_repeat)
        kwargs['num_fuse'] = num_invert_fuse
        kwargs['style'] = invert_style
        kwargs['act_cfg'] = invert_act_cfg
        self.down2top = nn.ModuleList([Down2TopFuseLayer(**kwargs)]*num_repeat)

    def forward(self, feats):
        for i in range(len(self.top2down)):
            feats = self.top2down[i](feats)
            feats = self.down2top[i](feats)
        if self.act_last:
            feats=[F.relu(feat) for feat in feats]
        return feats
