# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer, build_activation_layer
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.cnn.bricks.drop import DropPath




class ConvBlock(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 drop_path_rate=0.,
                 with_residual_conv=False,
                 norm_cfg=dict(type='BN', eps=1e-6),
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_cfg=None):
        super(ConvBlock, self).__init__(init_cfg=init_cfg)

        expansion = 4
        #mid_channels = out_channels // expansion

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.bn1 = build_norm_layer(norm_cfg, 64)[1]
        self.act1 = build_activation_layer(act_cfg)

        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.bn2 = build_norm_layer(norm_cfg, 64)[1]
        self.act2 = build_activation_layer(act_cfg)

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=96,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn3 = build_norm_layer(norm_cfg, 96)[1]
        self.act3 = build_activation_layer(act_cfg)

        self.conv4 = nn.Conv2d(
            in_channels=96,
            out_channels=96,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.bn4 = build_norm_layer(norm_cfg, 96)[1]
        self.act4 = build_activation_layer(act_cfg)

        self.conv5 = nn.Conv2d(
            in_channels=96,
            out_channels=192,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn5 = build_norm_layer(norm_cfg, 192)[1]
        self.act5 = build_activation_layer(act_cfg)

        self.conv6 = nn.Conv2d(
            in_channels=192,
            out_channels=192,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.bn6 = build_norm_layer(norm_cfg, 192)[1]
        self.act6 = build_activation_layer(act_cfg)

        self.conv7 = nn.Conv2d(
            in_channels=192,
            out_channels=384,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn7 = build_norm_layer(norm_cfg, 384)[1]
        self.act7 = build_activation_layer(act_cfg)

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    # def zero_init_last_bn(self):
    #     nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        identity = x

        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.act1(x1)
        temp = self.avgpool(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.act2(x2)

        x3 = x2+temp
        x3 = self.conv3(x3)
        x3 = self.bn3(x3)
        x3 = self.act3(x3)
        temp = self.avgpool(x3)

        x4 = self.conv4(x3)
        x4 = self.bn4(x4)
        x4 = self.act4(x4)

        x5 = x4 + temp
        x5 = self.conv5(x5)
        x5 = self.bn5(x5)
        x5 = self.act5(x5)
        temp = self.avgpool(x5)

        x6 = self.conv6(x5)
        x6 = self.bn6(x6)
        x6 = self.act6(x6)

        x7 = x6 + temp
        x7 = self.conv7(x7)
        x7 = self.bn7(x7)
        x7 = self.act7(x7)
        return x3,x5,x7


class ConToTans(BaseModule):
    """CNN feature maps -> Transformer patch embeddings."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 down_stride=4,
                 with_cls_token=True,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 init_cfg=None):
        super(ConToTans, self).__init__(init_cfg=init_cfg)
        self.down_stride = down_stride
        self.with_cls_token = with_cls_token

        self.conv_project = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(
            kernel_size=down_stride, stride=down_stride)

        self.ln = build_norm_layer(norm_cfg, out_channels)[1]
        self.act = build_activation_layer(act_cfg)

    def forward(self, x, in_channels, out_channels):
        x = self.conv_project(x).flatten(2).transpose(1, 2)  # [N, C, H, W]

        # x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)

        return x

import cv2 as cv
import numpy as np
img = cv.imread('../../data/train/1/236.jpg')
img = cv.resize(img,(256,256))
img_arr = np.array(img)
img_torch = torch.from_numpy(img_arr).reshape(1,img.shape[0],img.shape[1],3).transpose(3,2).transpose(1,2).to(torch.float32)
conv = ConvBlock(in_channels = 3, out_channels = 384)
x1,x2,x3 = conv(img_torch)














# class ConBranch(BaseModule):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  conv_stride=1,
#                  with_residual_conv=False,
#                  down_stride=4,
#                  with_cls_token=False,
#                  init_cfg = None):
#         super(ConBranch, self).__init__(init_cfg)

#         self.cnn_block = ConvBlock(
#             in_channels=in_channels,
#             out_channels=in_channels,
#             with_residual_conv=with_residual_conv,
#             stride=conv_stride
#         )

#         self.ConToTans = ConToTans(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             down_stride=down_stride,
#             with_cls_token=with_cls_token)

#     def forward(self, x):
#         x = self.cnn_block(x)
#         x = self.ConToTans(x)
#         return x


