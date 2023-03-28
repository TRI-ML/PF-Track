# ------------------------------------------------------------------------
# Copyright (c) 2023 toyota research instutute.
# ------------------------------------------------------------------------
""" Spatial-temporal Reasoning Module
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core.instances import Instances
from mmcv.cnn import Conv2d, Linear
from projects.mmdet3d_plugin.models.dense_heads.petr_head import pos2posemb3d
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.utils import build_transformer
from .utils import time_position_embedding, xyz_ego_transformation, normalize, denormalize


class SpatialTemporalReasoner(nn.Module):
    def __init__(self, 
                 history_reasoning=True,
                 future_reasoning=True,
                 embed_dims=256, 
                 hist_len=3, 
                 fut_len=4,
                 num_reg_fcs=2,
                 code_size=10,
                 num_classes=10,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 hist_temporal_transformer=None,
                 fut_temporal_transformer=None,
                 spatial_transformer=None):
        super(SpatialTemporalReasoner, self).__init__()

        self.embed_dims = embed_dims
        self.hist_len = hist_len
        self.fut_len = fut_len
        self.num_reg_fcs = num_reg_fcs
        self.pc_range = pc_range

        self.num_classes = num_classes
        self.code_size = code_size

        # If use history/future reasoning to improve the performance
        # affect initialization and behaviors
        self.history_reasoning = history_reasoning
        self.future_reasoning = future_reasoning

        # Transformer configurations
        self.hist_temporal_transformer = hist_temporal_transformer
        self.fut_temporal_transformer = fut_temporal_transformer
        self.spatial_transformer = spatial_transformer

        self.init_params_and_layers()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, track_instances):
        # 1. Prepare the spatial-temporal features
        track_instances = self.frame_shift(track_instances)

        # 2. History reasoning
        if self.history_reasoning:
            track_instances = self.forward_history_reasoning(track_instances)
            track_instances = self.forward_history_refine(track_instances)

        # 3. Future reasoning
        if self.future_reasoning:
            track_instances = self.forward_future_reasoning(track_instances)
            track_instances = self.forward_future_prediction(track_instances)
        return track_instances

    def forward_history_reasoning(self, track_instances: Instances):
        """Using history information to refine the current frame features
        """
        if len(track_instances) == 0:
            return track_instances
        
        valid_idxes = (track_instances.hist_padding_masks[:, -1] == 0)
        embed = track_instances.cache_query_feats[valid_idxes]

        if len(embed) == 0:
            return track_instances
        
        hist_embed = track_instances.hist_embeds[valid_idxes]
        hist_padding_mask = track_instances.hist_padding_masks[valid_idxes]

        ts_pe = time_position_embedding(hist_embed.shape[0], self.hist_len, 
                                        self.embed_dims, hist_embed.device)
        ts_pe = self.ts_query_embed(ts_pe)
        temporal_embed = self.hist_temporal_transformer(
            target=embed[:, None, :], 
            x=hist_embed, 
            query_embed=ts_pe[:, -1:, :],
            pos_embed=ts_pe,
            query_key_padding_mask=hist_padding_mask[:, -1:],
            key_padding_mask=hist_padding_mask)

        hist_pe = track_instances.cache_query_embeds[valid_idxes, None, :]
        spatial_embed = self.spatial_transformer(
            target=temporal_embed.transpose(0, 1),
            x=temporal_embed.transpose(0, 1),
            query_embed=hist_pe.transpose(0, 1),
            pos_embed=hist_pe.transpose(0, 1),
            query_key_padding_mask=hist_padding_mask[:, -1:].transpose(0, 1),
            key_padding_mask=hist_padding_mask[:, -1:].transpose(0, 1))[0]

        track_instances.cache_query_feats[valid_idxes] = spatial_embed.clone()
        track_instances.hist_embeds[valid_idxes, -1] = spatial_embed.clone().detach()
        return track_instances
    
    def forward_history_refine(self, track_instances: Instances):
        """Refine localization and classification"""
        if len(track_instances) == 0:
            return track_instances
        
        valid_idxes = (track_instances.hist_padding_masks[:, -1] == 0)
        embed = track_instances.cache_query_feats[valid_idxes]

        if len(embed) == 0:
            return track_instances
        
        """Classification"""
        logits = self.track_cls(track_instances.cache_query_feats[valid_idxes])
        # track_instances.hist_logits[valid_idxes, -1, :] = logits.clone()
        track_instances.cache_logits[valid_idxes] = logits.clone()
        track_instances.cache_scores = logits.sigmoid().max(dim=-1).values

        """Localization"""
        reference = inverse_sigmoid(track_instances.cache_reference_points[valid_idxes].clone())
        deltas = self.track_reg(track_instances.cache_query_feats[valid_idxes])
        deltas[..., [0, 1, 4]] += reference
        deltas[..., [0, 1, 4]] = deltas[..., [0, 1, 4]].sigmoid()

        track_instances.cache_reference_points[valid_idxes] = deltas[..., [0, 1, 4]].clone()
        # track_instances.hist_xyz[valid_idxes, -1, :] = deltas[..., [0, 1, 4]].clone()
        deltas[..., [0, 1, 4]] = denormalize(deltas[..., [0, 1, 4]], self.pc_range)
        track_instances.cache_bboxes[valid_idxes, :] = deltas
        # track_instances.hist_bboxes[valid_idxes, -1, :] = deltas.clone()
        return track_instances

    def forward_future_reasoning(self, track_instances: Instances):
        hist_embeds = track_instances.hist_embeds
        hist_padding_masks = track_instances.hist_padding_masks
        ts_pe = time_position_embedding(hist_embeds.shape[0], self.hist_len + self.fut_len, 
                                        self.embed_dims, hist_embeds.device)
        ts_pe = self.ts_query_embed(ts_pe)
        fut_embeds = self.fut_temporal_transformer(
            target=torch.zeros_like(ts_pe[:, self.hist_len:, :]),
            x=hist_embeds,
            query_embed=ts_pe[:, self.hist_len:],
            pos_embed=ts_pe[:, :self.hist_len],
            key_padding_mask=hist_padding_masks)
        track_instances.fut_embeds = fut_embeds
        return track_instances
    
    def forward_future_prediction(self, track_instances):
        """Predict the future motions"""
        motion_predictions = self.future_reg(track_instances.fut_embeds)
        track_instances.cache_motion_predictions = motion_predictions
        return track_instances

    def init_params_and_layers(self):
        # Modules for history reasoning
        if self.history_reasoning:
            # temporal transformer
            self.hist_temporal_transformer = build_transformer(self.hist_temporal_transformer)
            self.spatial_transformer = build_transformer(self.spatial_transformer)

            # classification refinement from history
            cls_branch = []
            for _ in range(self.num_reg_fcs):
                cls_branch.append(Linear(self.embed_dims, self.embed_dims))
                cls_branch.append(nn.LayerNorm(self.embed_dims))
                cls_branch.append(nn.ReLU(inplace=True))
            cls_branch.append(Linear(self.embed_dims, self.num_classes))
            self.track_cls = nn.Sequential(*cls_branch)
    
            # localization refinement from history
            reg_branch = []
            for _ in range(self.num_reg_fcs):
                reg_branch.append(Linear(self.embed_dims, self.embed_dims))
                reg_branch.append(nn.LayerNorm(self.embed_dims))
                reg_branch.append(nn.ReLU(inplace=True))
            reg_branch.append(Linear(self.embed_dims, self.code_size))
            self.track_reg = nn.Sequential(*reg_branch)
        
        # Modules for future reasoning
        if self.future_reasoning:
            # temporal transformer
            self.fut_temporal_transformer = build_transformer(self.fut_temporal_transformer)

            # regression head
            reg_branch = []
            for _ in range(self.num_reg_fcs):
                reg_branch.append(Linear(self.embed_dims, self.embed_dims))
                reg_branch.append(nn.LayerNorm(self.embed_dims))
                reg_branch.append(nn.ReLU(inplace=True))
            reg_branch.append(Linear(self.embed_dims, 3))
            self.future_reg = nn.Sequential(*reg_branch)
        
        if self.future_reasoning or self.history_reasoning:
            self.ts_query_embed = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
            )
        return
    
    def sync_pos_embedding(self, track_instances: Instances, mlp_embed: nn.Module = None):
        """Synchronize the positional embedding across all fields"""
        if mlp_embed is not None:
            track_instances.query_embeds = mlp_embed(pos2posemb3d(track_instances.reference_points))
            track_instances.hist_position_embeds = mlp_embed(pos2posemb3d(track_instances.hist_xyz))
        return track_instances
    
    def update_ego(self, track_instances: Instances, l2g0, l2g1):
        """Update the ego coordinates for reference points, hist_xyz, and fut_xyz of the track_instances
           Modify the centers of the bboxes at the same time
        Args:
            track_instances: objects
            l2g0: a [4x4] matrix for current frame lidar-to-global transformation 
            l2g1: a [4x4] matrix for target frame lidar-to-global transformation
        Return:
            transformed track_instances (inplace)
        """
        # TODO: orientation of the bounding boxes
        """1. Current states"""
        ref_points = track_instances.reference_points.clone()
        physical_ref_points = xyz_ego_transformation(ref_points, l2g0, l2g1, self.pc_range,
                                                     src_normalized=True, tgt_normalized=False)
        track_instances.bboxes[..., [0, 1, 4]] = physical_ref_points.clone()
        track_instances.reference_points = normalize(physical_ref_points, self.pc_range)
        
        """2. History states"""
        inst_num = len(track_instances)
        hist_ref_xyz = track_instances.hist_xyz.clone().view(inst_num * self.hist_len, 3)
        physical_hist_ref = xyz_ego_transformation(hist_ref_xyz, l2g0, l2g1, self.pc_range,
                                                   src_normalized=True, tgt_normalized=False)
        physical_hist_ref = physical_hist_ref.reshape(inst_num, self.hist_len, 3)
        track_instances.hist_bboxes[..., [0, 1, 4]] = physical_hist_ref
        track_instances.hist_xyz = normalize(physical_hist_ref, self.pc_range)
        
        """3. Future states"""
        inst_num = len(track_instances)
        fut_ref_xyz = track_instances.fut_xyz.clone().view(inst_num * self.fut_len, 3)
        physical_fut_ref = xyz_ego_transformation(fut_ref_xyz, l2g0, l2g1, self.pc_range,
                                                   src_normalized=True, tgt_normalized=False)
        physical_fut_ref = physical_fut_ref.reshape(inst_num, self.fut_len, 3)
        track_instances.fut_bboxes[..., [0, 1, 4]] = physical_fut_ref
        track_instances.fut_xyz = normalize(physical_fut_ref, self.pc_range)

        return track_instances
    
    def update_reference_points(self, track_instances, time_deltas, use_prediction=True, tracking=False):
        """Update the reference points according to the motion prediction
           Used for next frame
        """
        if use_prediction:
            # inference mode, use multi-step forecasting to modify reference points
            if tracking:
                motions = track_instances.motion_predictions[:, 0, :2].clone()
                reference_points = track_instances.reference_points.clone()
                motions[:, 0] /= (self.pc_range[3] - self.pc_range[0])
                motions[:, 1] /= (self.pc_range[4] - self.pc_range[1])
                reference_points[..., :2] += motions.clone().detach()
                track_instances.reference_points = reference_points
            # training mode, single-step prediction
            else:
                track_instances.reference_points = track_instances.fut_xyz[:, 0, :].clone()
                track_instances.bboxes = track_instances.fut_bboxes[:, 0, :].clone()
        else:
            velos = track_instances.bboxes[..., 7:9].clone()
            reference_points = track_instances.reference_points.clone()
            velos[:, 0] /= (self.pc_range[3] - self.pc_range[0])
            velos[:, 1] /= (self.pc_range[4] - self.pc_range[1])
            reference_points[..., :2] += (velos * time_deltas).clone().detach()
            track_instances.reference_points = reference_points
        return track_instances
    
    def frame_shift(self, track_instances: Instances):
        """Shift the information for the newest frame before spatial-temporal reasoning happens. 
           Pay attention to the order.
        """
        device = track_instances.query_feats.device
        
        """1. History reasoining"""
        # embeds
        track_instances.hist_embeds = track_instances.hist_embeds.clone()
        track_instances.hist_embeds = torch.cat((
            track_instances.hist_embeds[:, 1:, :], track_instances.cache_query_feats[:, None, :]), dim=1)
        # padding masks
        track_instances.hist_padding_masks = torch.cat((
            track_instances.hist_padding_masks[:, 1:], 
            torch.zeros((len(track_instances), 1), dtype=torch.bool, device=device)), 
            dim=1)
        # positions
        track_instances.hist_xyz = torch.cat((
            track_instances.hist_xyz[:, 1:, :], track_instances.cache_reference_points[:, None, :]), dim=1)
        # positional embeds
        track_instances.hist_position_embeds = torch.cat((
            track_instances.hist_position_embeds[:, 1:, :], track_instances.cache_query_embeds[:, None, :]), dim=1)
        # bboxes
        track_instances.hist_bboxes = torch.cat((
            track_instances.hist_bboxes[:, 1:, :], track_instances.cache_bboxes[:, None, :]), dim=1)
        # logits
        track_instances.hist_logits = torch.cat((
            track_instances.hist_logits[:, 1:, :], track_instances.cache_logits[:, None, :]), dim=1)
        # scores
        track_instances.hist_scores = torch.cat((
            track_instances.hist_scores[:, 1:], track_instances.cache_scores[:, None]), dim=1)
        
        """2. Temporarily load motion predicted results as final results"""
        if self.future_reasoning:
            track_instances.reference_points = track_instances.fut_xyz[:, 0, :].clone()
            track_instances.bboxes = track_instances.fut_bboxes[:, 0, :].clone()
            track_instances.scores = track_instances.fut_scores[:, 0].clone() + 0.01
            track_instances.logits = track_instances.fut_logits[:, 0, :].clone()

        """3. Future reasoning"""
        # TODO: shift the future information, useful for occlusion handling
        track_instances.motion_predictions = torch.cat((
            track_instances.motion_predictions[:, 1:, :], torch.zeros_like(track_instances.motion_predictions[:, 0:1, :])), dim=1)
        track_instances.fut_embeds = torch.cat((
            track_instances.fut_embeds[:, 1:, :], torch.zeros_like(track_instances.fut_embeds[:, 0:1, :])), dim=1)
        track_instances.fut_padding_masks = torch.cat((
            track_instances.fut_padding_masks[:, 1:], torch.ones_like(track_instances.fut_padding_masks[:, 0:1]).bool()), dim=1)
        track_instances.fut_xyz = torch.cat((
            track_instances.fut_xyz[:, 1:, :], torch.zeros_like(track_instances.fut_xyz[:, 0:1, :])), dim=1)
        track_instances.fut_position_embeds = torch.cat((
            track_instances.fut_position_embeds[:, 1:, :], torch.zeros_like(track_instances.fut_position_embeds[:, 0:1, :])), dim=1)
        track_instances.fut_bboxes = torch.cat((
            track_instances.fut_bboxes[:, 1:, :], torch.zeros_like(track_instances.fut_bboxes[:, 0:1, :])), dim=1)
        track_instances.fut_logits = torch.cat((
            track_instances.fut_logits[:, 1:, :], torch.zeros_like(track_instances.fut_logits[:, 0:1, :])), dim=1)
        track_instances.fut_scores = torch.cat((
            track_instances.fut_scores[:, 1:], torch.zeros_like(track_instances.fut_scores[:, 0:1])), dim=1)
        return track_instances