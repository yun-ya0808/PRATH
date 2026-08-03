# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
AnchorDETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .misc import (NestedTensor, nested_tensor_from_tensor_list)

from .backbone import build_backbone
from .transformer import build_transformer
import copy
from bottleneck_transformer_pytorch import BottleStack
# from .CBAM import CBAMBlock
from .CoordAttention import CoordAtt


class AnchorDETR(nn.Module):
    """ This is the AnchorDETR module that performs object detection """

    def __init__(self, backbone, transformer, num_feature_levels, aux_loss=True):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_classes = 16
        self.num_feature_levels = num_feature_levels
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            # cbam = []
            # catt = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                if _ == 0:
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))

                    # cbam.append(CBAMBlock(channel=hidden_dim, reduction=16, kernel_size=15))
                    # catt.append(CoordAtt(hidden_dim, hidden_dim, reduction=16))
                elif _ == 1:
                    input_proj_list.append(nn.Sequential(  # 是不是应该先卷积，后上注意力呢？
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))

                    # cbam.append(CBAMBlock(channel=hidden_dim, reduction=16, kernel_size=15))
                    # catt.append(CoordAtt(hidden_dim, hidden_dim, reduction=16))
                else:
                    input_proj_list.append(nn.Sequential(  # 是不是应该先卷积，后上注意力呢？
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))

                    # cbam.append(CBAMBlock(channel=hidden_dim, reduction=16, kernel_size=15))
                    # catt.append(CoordAtt(hidden_dim, hidden_dim, reduction=16))
            self.input_proj = nn.ModuleList(input_proj_list)
            # self.cbam = nn.ModuleList(cbam)
            # self.catt = nn.ModuleList(catt)
            # self.cbam = CBAMBlock(channel=hidden_dim, reduction=16, kernel_size=15)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
            # self.cbam = nn.ModuleList([CBAMBlock(channel=hidden_dim, reduction=16, kernel_size=15)])
        self.backbone = backbone
        self.aux_loss = aux_loss

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features = self.backbone(samples)  # [2,2048,18,24]

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            # srcs.append(self.catt[l](self.input_proj[l](src)))
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        # srcs = torch.cat(srcs, dim=1)  # [bs,1,256,18,24]
        # srcs = self.cbam(srcs)  # 先减少通道数，然后堆叠在一起，然后再使用通道注意力

        # 三级特征图直接加在一起
        srcs = srcs[0] + srcs[1] + srcs[2]
        # CBAM 的对照试验，感觉CBAM应该没有CA效果好，因为CA加了位置信息，但是也不好说，反正这个到时候效果更好可以直接换过去就行
        # srcs = self.cbam(srcs)

        outputs_class, outputs_coord = self.transformer(srcs, masks)

        out = {'pred_logits': outputs_class[-1], 'pred_coords': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_coords': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


def get_pose_net(cfg, is_train, **kwargs):
    backbone = build_backbone(cfg)
    transformer = build_transformer(cfg)
    model = AnchorDETR(
        backbone,
        transformer,
        num_feature_levels=cfg.MODEL.NUM_FEATURE_LEVELS,
        aux_loss=cfg.MODEL.EXTRA.AUX_LOSS
    )

    return model
