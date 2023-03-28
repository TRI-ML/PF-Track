# ------------------------------------------------------------------------
# Copyright (c) 2022 toyota research institute
# ------------------------------------------------------------------------
# Modified from PETR (https://github.com/megvii-research/PETR)
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import math
import warnings
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.cnn.bricks.drop import build_dropout
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, xavier_init)
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import (ATTENTION,TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
                        to_2tuple)
import copy
import torch.utils.checkpoint as cp


@TRANSFORMER.register_module()
class PETRTrackingTransformer(BaseModule):
    """Implements the DETR transformer.
    Adding a field of target to denote the query features.
    If target is None, the behavior is identical to PETRTransformer.
    """
    def __init__(self, encoder=None, decoder=None, init_cfg=None, cross=False):
        super(PETRTrackingTransformer, self).__init__(init_cfg=init_cfg)
        if encoder is not None:
            self.encoder = build_transformer_layer_sequence(encoder)
        else:
            self.encoder = None
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.cross = cross
    
    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True
    
    def forward(self, target, x, mask, query_embed, pos_embed, reg_branch=None):
        bs, n, c, h, w = x.shape
        memory = x.permute(1, 3, 4, 0, 2).reshape(-1, bs, c) # [bs, n, c, h, w] -> [n*h*w, bs, c]
        pos_embed = pos_embed.permute(1, 3, 4, 0, 2).reshape(-1, bs, c) # [bs, n, c, h, w] -> [n*h*w, bs, c]
        query_embed = query_embed.unsqueeze(1).repeat(
            1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]
        target = target.unsqueeze(1).repeat(
            1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]
        mask = mask.view(bs, -1)  # [bs, n, h, w] -> [bs, n*h*w]

        # out_dec: [num_layers, num_query, bs, dim]
        out_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask,
            reg_branch=reg_branch,
            )
        out_dec = out_dec.transpose(1, 2)
        memory = memory.reshape(n, h, w, bs, c).permute(3, 0, 4, 1, 2)
        return  out_dec, memory
