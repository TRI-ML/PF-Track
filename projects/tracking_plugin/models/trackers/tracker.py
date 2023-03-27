# ------------------------------------------------------------------------
# Copyright (c) 2023 toyota research instutute.
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from contextlib import nullcontext
from mmcv.cnn import Conv2d, Linear, build_activation_layer, bias_init_with_prob
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet.models import DETECTORS
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet.models import build_loss
from copy import deepcopy
from mmcv.runner import force_fp32, auto_fp16
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.dense_heads.petr_head import pos2posemb3d
from projects.tracking_plugin.core.instances import Instances
from .runtime_tracker import RunTimeTracker
from .spatial_temporal_reason import SpatialTemporalReasoner
from .utils import time_position_embedding, xyz_ego_transformation, normalize, denormalize


@DETECTORS.register_module()
class Cam3DTracker(MVXTwoStageDetector):
    def __init__(self,
                 num_classes=10,
                 num_query=100,
                 tracking=True,
                 train_backbone=True,
                 if_update_ego=True,
                 motion_prediction=True,
                 motion_prediction_ref_update=True,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                 spatial_temporal_reason=None,
                 runtime_tracker=None,
                 loss=None,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Cam3DTracker, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                                           pts_middle_encoder, pts_fusion_layer,
                                           img_backbone, pts_backbone, img_neck, pts_neck,
                                           pts_bbox_head, img_roi_head, img_rpn_head,
                                           train_cfg, test_cfg, pretrained)
        self.num_classes = num_classes
        self.num_query = num_query
        self.embed_dims = 256
        self.tracking = tracking
        self.train_backbone = train_backbone
        self.if_update_ego = if_update_ego
        self.motion_prediction = motion_prediction
        self.motion_prediction_ref_update=motion_prediction_ref_update
        self.pc_range = pc_range
        self.position_range = position_range
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.criterion = build_loss(loss)

        # spatial-temporal reasoning
        self.STReasoner = SpatialTemporalReasoner(**spatial_temporal_reason)
        self.hist_len = self.STReasoner.hist_len
        self.fut_len = self.STReasoner.fut_len

        self.init_params_and_layers()

        # Inference time tracker
        self.runtime_tracker = RunTimeTracker(**runtime_tracker)
    
    def generate_empty_instance(self):
        """Generate empty instance slots at the beginning of tracking"""
        track_instances = Instances((1, 1))
        device = self.reference_points.weight.device

        """Detection queries"""
        # reference points, query embeds, and query targets (features)
        reference_points = self.reference_points.weight
        query_embeds = self.query_embedding(pos2posemb3d(reference_points))
        track_instances.reference_points = reference_points.clone()
        track_instances.query_embeds = query_embeds.clone()
        if self.tracking:
            track_instances.query_feats = self.query_feat_embedding.weight.clone()
        else:
            track_instances.query_feats = torch.zeros_like(query_embeds)

        """Tracking information"""
        # id for the tracks
        track_instances.obj_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device)
        # matched gt indexes, for loss computation
        track_instances.matched_gt_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device)
        # life cycle management
        track_instances.disappear_time = torch.zeros(
            (len(track_instances), ), dtype=torch.long, device=device)
        track_instances.track_query_mask = torch.zeros(
            (len(track_instances), ), dtype=torch.bool, device=device)

        """Current frame information"""
        # classification scores
        track_instances.logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device)
        # bounding boxes
        track_instances.bboxes = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device)
        # track scores, normally the scores for the highest class
        track_instances.scores = torch.zeros(
            (len(track_instances)), dtype=torch.float, device=device)
        # motion prediction, not normalized
        track_instances.motion_predictions = torch.zeros(
            (len(track_instances), self.fut_len, 3), dtype=torch.float, device=device)
        
        """Cache for current frame information, loading temporary data for spatial-temporal reasoining"""
        track_instances.cache_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device)
        track_instances.cache_bboxes = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device)
        track_instances.cache_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device)
        track_instances.cache_reference_points = reference_points.clone()
        track_instances.cache_query_embeds = query_embeds.clone()
        if self.tracking:
            track_instances.cache_query_feats = self.query_feat_embedding.weight.clone()
        else:
            track_instances.cache_query_feats = torch.zeros_like(query_embeds)
        track_instances.cache_motion_predictions = torch.zeros_like(track_instances.motion_predictions)

        """History Reasoning"""
        # embeddings
        track_instances.hist_embeds = torch.zeros(
            (len(track_instances), self.hist_len, self.embed_dims), dtype=torch.float32, device=device)
        # padding mask, follow MultiHeadAttention, 1 indicates padded
        track_instances.hist_padding_masks = torch.ones(
            (len(track_instances), self.hist_len), dtype=torch.bool, device=device)
        # positions
        track_instances.hist_xyz = torch.zeros(
            (len(track_instances), self.hist_len, 3), dtype=torch.float, device=device)
        # positional embeds
        track_instances.hist_position_embeds = torch.zeros(
            (len(track_instances), self.hist_len, self.embed_dims), dtype=torch.float32, device=device)
        # bboxes
        track_instances.hist_bboxes = torch.zeros(
            (len(track_instances), self.hist_len, 10), dtype=torch.float, device=device)
        # logits
        track_instances.hist_logits = torch.zeros(
            (len(track_instances), self.hist_len, self.num_classes), dtype=torch.float, device=device)
        # scores
        track_instances.hist_scores = torch.zeros(
            (len(track_instances), self.hist_len), dtype=torch.float, device=device)

        """Future Reasoning"""
        # embeddings
        track_instances.fut_embeds = torch.zeros(
            (len(track_instances), self.fut_len, self.embed_dims), dtype=torch.float32, device=device)
        # padding mask, follow MultiHeadAttention, 1 indicates padded
        track_instances.fut_padding_masks = torch.ones(
            (len(track_instances), self.fut_len), dtype=torch.bool, device=device)
        # positions
        track_instances.fut_xyz = torch.zeros(
            (len(track_instances), self.fut_len, 3), dtype=torch.float, device=device)
        # positional embeds
        track_instances.fut_position_embeds = torch.zeros(
            (len(track_instances), self.fut_len, self.embed_dims), dtype=torch.float32, device=device)
        # bboxes
        track_instances.fut_bboxes = torch.zeros(
            (len(track_instances), self.fut_len, 10), dtype=torch.float, device=device)
        # logits
        track_instances.fut_logits = torch.zeros(
            (len(track_instances), self.fut_len, self.num_classes), dtype=torch.float, device=device)
        # scores
        track_instances.fut_scores = torch.zeros(
            (len(track_instances), self.fut_len), dtype=torch.float, device=device)
        return track_instances
    
    def update_ego(self, track_instances, l2g0, l2g1):
        """Update the ego coordinates for reference points, hist_xyz, and fut_xyz of the track_instances
           Modify the centers of the bboxes at the same time
        """
        track_instances = self.STReasoner.update_ego(track_instances, l2g0, l2g1)
        return track_instances
    
    def update_reference_points(self, track_instances, time_delta=None, use_prediction=True, tracking=False):
        """Update the reference points according to the motion prediction/velocities
        """
        track_instances = self.STReasoner.update_reference_points(
            track_instances, time_delta, use_prediction, tracking)
        return track_instances
    
    def load_detection_output_into_cache(self, track_instances: Instances, out):
        """ Load output of the detection head into the track_instances cache (inplace)
        """
        query_feats = out.pop('query_feats')
        query_reference_points = out.pop('reference_points')
        with torch.no_grad():
            track_scores = out['all_cls_scores'][-1, 0, :].sigmoid().max(dim=-1).values
        track_instances.cache_scores = track_scores.clone()
        track_instances.cache_logits = out['all_cls_scores'][-1, 0].clone()
        track_instances.cache_query_feats = query_feats[0].clone()
        track_instances.cache_reference_points = query_reference_points[0].clone()
        track_instances.cache_bboxes = out['all_bbox_preds'][-1, 0].clone()
        track_instances.cache_query_embeds = self.query_embedding(pos2posemb3d(track_instances.cache_reference_points))
        return track_instances
    
    def forward_loss_prediction(self, 
                                frame_idx,
                                loss_dict,
                                active_track_instances,
                                gt_trajs,
                                gt_traj_masks,
                                instance_inds,):
        active_gt_trajs, active_gt_traj_masks = list(), list()
        obj_idx_to_gt_idx = {obj_idx: gt_idx for gt_idx, obj_idx in enumerate(
            instance_inds[0].detach().cpu().numpy().tolist())}

        active_gt_trajs = torch.ones_like(active_track_instances.motion_predictions)
        active_gt_trajs[..., -1] = 0.0
        active_gt_traj_masks = torch.zeros_like(active_gt_trajs)[..., 0]
        for track_idx, id in enumerate(active_track_instances.obj_idxes):
            cpu_id = id.cpu().numpy().tolist()
            if cpu_id not in obj_idx_to_gt_idx.keys():
                continue
            index = obj_idx_to_gt_idx[cpu_id]
            traj = gt_trajs[index:index+1, :self.fut_len + 1, :]
            gt_motion = traj[:, torch.arange(1, self.fut_len + 1)] - traj[:, torch.arange(0, self.fut_len)]
            active_gt_trajs[track_idx: track_idx + 1] = gt_motion
            active_gt_traj_masks[track_idx: track_idx + 1] = \
                gt_traj_masks[index: index+1, 1: self.fut_len + 1] * gt_traj_masks[index: index+1, : self.fut_len]
        
        loss_dict = self.criterion.loss_prediction(frame_idx,
                                                   loss_dict,
                                                   active_gt_trajs[..., :2],
                                                   active_gt_traj_masks,
                                                   active_track_instances.cache_motion_predictions[..., :2])
        return loss_dict
    
    def frame_summarization(self, track_instances, tracking=False):
        """ Load the results after spatial-temporal reasoning into track instances
        """
        # inference mode
        if tracking:
            active_mask = (track_instances.cache_scores >= self.runtime_tracker.record_threshold)
        # training mode
        else:
            track_instances.bboxes = track_instances.cache_bboxes.clone()
            track_instances.logits = track_instances.cache_logits.clone()
            track_instances.scores = track_instances.cache_scores.clone()
            active_mask = (track_instances.cache_scores >= 0.0)

        track_instances.query_feats[active_mask] = track_instances.cache_query_feats[active_mask]
        track_instances.query_embeds[active_mask] = track_instances.cache_query_embeds[active_mask]
        track_instances.logits[active_mask] = track_instances.cache_logits[active_mask]
        track_instances.scores[active_mask] = track_instances.cache_scores[active_mask]
        track_instances.motion_predictions[active_mask] = track_instances.cache_motion_predictions[active_mask]
        track_instances.bboxes[active_mask] = track_instances.cache_bboxes[active_mask]
        track_instances.reference_points[active_mask] = track_instances.cache_reference_points[active_mask]

        # TODO: generate future bounding boxes, reference points, scores
        if self.STReasoner.future_reasoning:
            motion_predictions = track_instances.motion_predictions[active_mask]
            track_instances.fut_xyz[active_mask] = track_instances.reference_points[active_mask].clone()[:, None, :].repeat(1, self.fut_len, 1)
            track_instances.fut_bboxes[active_mask] = track_instances.bboxes[active_mask].clone()[:, None, :].repeat(1, self.fut_len, 1)
            motion_add = torch.cumsum(motion_predictions.clone().detach(), dim=1)
            motion_add_normalized = motion_add.clone()
            motion_add_normalized[..., 0] /= (self.pc_range[3] - self.pc_range[0])
            motion_add_normalized[..., 1] /= (self.pc_range[4] - self.pc_range[1])
            track_instances.fut_xyz[active_mask, :, 0] += motion_add_normalized[..., 0]
            track_instances.fut_xyz[active_mask, :, 1] += motion_add_normalized[..., 1]
            track_instances.fut_bboxes[active_mask, :, 0] += motion_add[..., 0]
            track_instances.fut_bboxes[active_mask, :, 1] += motion_add[..., 1]
        return track_instances
   
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      instance_inds=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      l2g=None,
                      timestamp=None,
                      img_depth=None,
                      img_mask=None,
                      **kwargs):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                For each sample, the format is Dict(key: contents of the timestamps)
                Defaults to None. For each field, its shape is [T * NumCam * ContentLength]
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None. Number same as batch size.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, T, Num_Cam, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
            l2g_r_mat (list[Tensor]). Lidar to global transformation, shape [T, 3, 3]
            l2g_t (list[Tensor]). Lidar to global rotation
                points @ R_Mat + T
            timestamp (list). Timestamp of the frames
        Returns:
            dict: Losses of different branches.
        """
        batch_size, num_frame, num_cam = img.shape[0], img.shape[1], img.shape[2]

        # Image features, one clip at a time for checkpoint usages
        img_feats = self.extract_clip_imgs_feats(img_metas=img_metas, img=img)

        # transform labels to a temporal frame-first sense
        ff_gt_bboxes_list, ff_gt_labels_list, ff_instance_ids = list(), list(), list()
        for i in range(num_frame):
            ff_gt_bboxes_list.append([gt_bboxes_3d[j][i] for j in range(batch_size)])
            ff_gt_labels_list.append([gt_labels_3d[j][i] for j in range(batch_size)])
            ff_instance_ids.append([instance_inds[j][i] for j in range(batch_size)])
        
        # Empty the runtime_tracker
        # Use PETR head to decode the bounding boxes on every frame
        outs = list()
        next_frame_track_instances = self.generate_empty_instance()
        img_metas_keys = img_metas[0].keys()

        # Running over all the frames one by one
        self.runtime_tracker.empty()
        for frame_idx in range(num_frame):
            img_metas_single_frame = list()
            for batch_idx in range(batch_size):
                img_metas_single_sample = {key: img_metas[batch_idx][key][frame_idx] for key in img_metas_keys}
                img_metas_single_frame.append(img_metas_single_sample)
            
            # PETR detection head
            track_instances = next_frame_track_instances
            out = self.pts_bbox_head(img_feats[frame_idx], img_metas_single_frame, 
                                     track_instances.query_feats, track_instances.query_embeds, 
                                     track_instances.reference_points)

            # 1. Record the information into the track instances cache
            track_instances = self.load_detection_output_into_cache(track_instances, out)
            out['track_instances'] = track_instances
            out['points'] = points[0][frame_idx]

            # 2. Loss computation for the detection
            out['loss_dict'] = self.criterion.loss_single_frame(frame_idx, 
                                                                ff_gt_bboxes_list[frame_idx],
                                                                ff_gt_labels_list[frame_idx],
                                                                ff_instance_ids[frame_idx],
                                                                out,
                                                                None)

            # 3. Spatial-temporal reasoning
            track_instances = self.STReasoner(track_instances)

            if self.STReasoner.history_reasoning:
                out['loss_dict'] = self.criterion.loss_mem_bank(frame_idx,
                                                                out['loss_dict'],
                                                                ff_gt_bboxes_list[frame_idx],
                                                                ff_gt_labels_list[frame_idx],
                                                                ff_instance_ids[frame_idx],
                                                                track_instances)
            
            if self.STReasoner.future_reasoning:
                active_mask = (track_instances.obj_idxes >= 0)
                out['loss_dict'] = self.forward_loss_prediction(frame_idx,
                                                                out['loss_dict'],
                                                                track_instances[active_mask],
                                                                kwargs['gt_forecasting_locs'][0][frame_idx],
                                                                kwargs['gt_forecasting_masks'][0][frame_idx],
                                                                ff_instance_ids[frame_idx])

            # 4. Prepare for next frame
            track_instances = self.frame_summarization(track_instances, tracking=False)
            active_mask = self.runtime_tracker.get_active_mask(track_instances, training=True)
            track_instances.track_query_mask[active_mask] = True
            active_track_instances = track_instances[active_mask]
            if self.motion_prediction and frame_idx < num_frame - 1:
                time_delta = timestamp[frame_idx + 1] - timestamp[frame_idx]
                active_track_instances = self.update_reference_points(active_track_instances,
                                                                      time_delta,
                                                                      use_prediction=self.motion_prediction_ref_update,
                                                                      tracking=False)
            if self.if_update_ego and frame_idx < num_frame - 1:
                active_track_instances = self.update_ego(active_track_instances, 
                                                         l2g[0][frame_idx], l2g[0][frame_idx + 1])
            if frame_idx < num_frame - 1:
                active_track_instances = self.STReasoner.sync_pos_embedding(active_track_instances, self.query_embedding)
            
            empty_track_instances = self.generate_empty_instance()
            next_frame_track_instances = Instances.cat([empty_track_instances, active_track_instances])
            self.runtime_tracker.frame_index += 1
            outs.append(out)
        losses = self.criterion(outs)
        self.runtime_tracker.empty()
        return losses
    
    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img=None,
                     proposals=None,
                     l2g=None,
                     timestamp=None,
                     img_depth=None,
                     img_mask=None,
                     rescale=False,
                     **kwargs):
        """ This function is not used for MOT, so I haven't paid attention to this.
        """
        batch_size, num_frame, num_cam = img.shape[0], img.shape[1], img.shape[2]

        # Image features
        img_feats = self.extract_clip_imgs_feats(img_metas=img_metas, img=img)

        outs = list()
        next_frame_track_instances = self.generate_empty_instance()
        img_metas_keys = img_metas[0].keys()
        self.runtime_tracker.empty()
        for frame_idx in range(num_frame):
            img_metas_single_frame = list()
            for batch_idx in range(batch_size):
                img_metas_single_sample = {key: img_metas[batch_idx][key][frame_idx] for key in img_metas_keys}
                img_metas_single_frame.append(img_metas_single_sample)
            
            # PETR detection head
            track_instances = next_frame_track_instances
            out = self.pts_bbox_head(img_feats[frame_idx], img_metas_single_frame, 
                                     track_instances.query_feats, track_instances.query_embeds, 
                                     track_instances.reference_points)

            # 1. Record the information into the track instances cache
            track_instances = self.load_detection_output_into_cache(track_instances, out)
            out['track_instances'] = track_instances

            # 2. Spatial-temporal Reasoning
            track_instances = self.STReasoner(track_instances)
            track_instances = self.frame_summarization(track_instances)
            out['all_cls_scores'][-1][0, :] = track_instances.logits
            out['all_bbox_preds'][-1][0, :] = track_instances.bboxes

            if self.STReasoner.future_reasoning:
                # motion forecasting has the shape of [num_query, T, 2]
                out['all_motion_forecasting'] = track_instances.motion_predictions.clone()
            else:
                out['all_motion_forecasting'] = None
            
            active_mask = (track_instances.scores >= self.runtime_tracker.threshold)
            active_track_instances = track_instances[active_mask]
            
            if self.if_update_ego and frame_idx < num_frame - 1:
                active_track_instances = self.update_ego(active_track_instances, 
                                                         l2g[0][frame_idx], l2g[0][frame_idx + 1])
            if self.motion_prediction and frame_idx < num_frame - 1:
                time_delta = timestamp[frame_idx + 1] - timestamp[frame_idx]
                active_track_instances = self.update_reference_points(active_track_instances,
                                                                      time_delta,
                                                                      use_prediction=self.motion_prediction_ref_update)
            if frame_idx < num_frame - 1:
                active_track_instances = self.STReasoner.sync_pos_embedding(active_track_instances, self.query_embedding)

            empty_track_instances = self.generate_empty_instance()
            next_frame_track_instances = Instances.cat([empty_track_instances, active_track_instances])
            self.runtime_tracker.frame_index += 1

            # if frame_idx == 0:
            if frame_idx == num_frame - 1:
                bbox_list = self.pts_bbox_head.get_bboxes(out, img_metas_single_frame, rescale=rescale, tracking=False)
        
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels, _, _, _ in bbox_list
        ]
        return bbox_results
    
    def forward_track(self,
                      points=None,
                      img_metas=None,
                      img=None,
                      proposals=None,
                      l2g_r_mat=None,
                      l2g_t=None,
                      l2g=None,
                      timestamp=None,
                      img_depth=None,
                      img_mask=None,
                      rescale=False,
                      **kwargs):
        batch_size, num_frame, num_cam = img.shape[0], img.shape[1], img.shape[2]
        
        # backbone images
        img_feats = self.extract_clip_imgs_feats(img_metas=img_metas, img=img)
        
        # image metas
        all_img_metas = list()
        img_metas_keys = img_metas[0].keys()
        for i in range(batch_size):
            img_metas_single_sample = dict()
            for key in img_metas_keys:
                # join the fields of every timestamp
                if type(img_metas[i][key][0]) == list:
                    contents = deepcopy(img_metas[i][key][0])
                    for j in range(1, num_frame):
                        contents += img_metas[i][key][j]
                # pick the representative one
                else:
                    contents = deepcopy(img_metas[i][key][0])
            img_metas_single_sample[key] = contents
            all_img_metas.append(img_metas_single_sample)
        
        # new sequence
        if self.runtime_tracker.timestamp is None or abs(timestamp[0] - self.runtime_tracker.timestamp) > 10:
            self.runtime_tracker.timestamp = timestamp[0]
            self.runtime_tracker.current_seq += 1
            self.runtime_tracker.track_instances = None
            self.runtime_tracker.current_id = 0
            self.runtime_tracker.l2g = None
            self.runtime_tracker.time_delta = 0
            self.runtime_tracker.frame_index = 0
        self.runtime_tracker.time_delta = timestamp[0] - self.runtime_tracker.timestamp
        self.runtime_tracker.timestamp = timestamp[0]
        
        # processing the queries from t-1
        prev_active_track_instances = self.runtime_tracker.track_instances
        for frame_idx in range(num_frame):
            img_metas_single_frame = list()
            for batch_idx in range(batch_size):
                img_metas_single_sample = {key: img_metas[batch_idx][key][frame_idx] for key in img_metas_keys}
                img_metas_single_frame.append(img_metas_single_sample)            

            # 1. Update the information of previous active tracks
            if prev_active_track_instances is None:
                track_instances = self.generate_empty_instance()
            else:
                if self.motion_prediction:
                    time_delta = self.runtime_tracker.time_delta
                    prev_active_track_instances = self.update_reference_points(prev_active_track_instances,
                                                                               time_delta,
                                                                               use_prediction=self.motion_prediction_ref_update,
                                                                               tracking=True)
                if self.if_update_ego:
                    prev_active_track_instances = self.update_ego(prev_active_track_instances,
                                                                  self.runtime_tracker.l2g, l2g[0][frame_idx])
                prev_active_track_instances = self.STReasoner.sync_pos_embedding(prev_active_track_instances, self.query_embedding)
                track_instances = Instances.cat([self.generate_empty_instance(), prev_active_track_instances])
            
            self.runtime_tracker.l2g = l2g[0][frame_idx]
            self.runtime_tracker.timestamp = timestamp[0]

            # 2. PETR detection head
            out = self.pts_bbox_head(img_feats[frame_idx], img_metas_single_frame, 
                                     track_instances.query_feats, track_instances.query_embeds, 
                                     track_instances.reference_points)

            # 3. Record the information into the track instances cache
            track_instances = self.load_detection_output_into_cache(track_instances, out)
            out['track_instances'] = track_instances

            # 4. Spatial-temporal Reasoning
            self.STReasoner(track_instances)
            track_instances = self.frame_summarization(track_instances, tracking=True)
            out['all_cls_scores'][-1][0, :] = track_instances.logits
            out['all_bbox_preds'][-1][0, :] = track_instances.bboxes

            if self.STReasoner.future_reasoning:
                # motion forecasting has the shape of [num_query, T, 2]
                out['all_motion_forecasting'] = track_instances.motion_predictions.clone()
            else:
                out['all_motion_forecasting'] = None

            # 5. Track class filtering: before decoding bboxes, only leave the objects under tracking categories
            max_cat = torch.argmax(out['all_cls_scores'][-1, 0, :].sigmoid(), dim=-1)
            related_cat_mask = (max_cat < 7) # we set the first 7 classes as the tracking classes of nuscenes
            track_instances = track_instances[related_cat_mask]
            out['all_cls_scores'] = out['all_cls_scores'][:, :, related_cat_mask, :]
            out['all_bbox_preds'] = out['all_bbox_preds'][:, :, related_cat_mask, :]
            if out['all_motion_forecasting'] is not None:
                out['all_motion_forecasting'] = out['all_motion_forecasting'][related_cat_mask, ...]

            # 6. assign ids
            active_mask = (track_instances.scores > self.runtime_tracker.threshold)
            for i in range(len(track_instances)):
                if track_instances.obj_idxes[i] < 0:
                    track_instances.obj_idxes[i] = self.runtime_tracker.current_id 
                    self.runtime_tracker.current_id += 1
                    if active_mask[i]:
                        track_instances.track_query_mask[i] = True
            out['track_instances'] = track_instances

            # 7. Prepare for the next frame and output
            score_mask = (track_instances.scores > self.runtime_tracker.output_threshold)
            out['all_masks'] = score_mask.clone()

            bbox_list = self.pts_bbox_head.get_bboxes(out, img_metas_single_frame, rescale=rescale, tracking=True)
            # self.runtime_tracker.update_active_tracks(active_track_instances)
            self.runtime_tracker.update_active_tracks(track_instances, active_mask)
            
            # each time, only run one frame
            self.runtime_tracker.frame_index += 1
            break
        
        bbox_results = [
            track_bbox3d2result(bboxes, scores, labels, obj_idxes, track_scores, forecasting)
            for bboxes, scores, labels, obj_idxes, track_scores, forecasting in bbox_list
        ]
        bbox_results[0]['track_ids'] = [f'{self.runtime_tracker.current_seq}-{i}' for i in bbox_results[0]['track_ids'].long().cpu().numpy().tolist()]
        return bbox_results
    
    def extract_clip_imgs_feats(self, img_metas, img):
        """Extract the features of multi-frame images
        Args:
            img_metas (list[dict], optional): Meta information of each sample.
                For each sample, the format is Dict(key: contents of the timestamps)
                Defaults to None. For each field, its shape is [T * NumCam * ContentLength]
            img (torch.Tensor optional): Images of each sample with shape
                (N, T, Num_Cam, C, H, W). Defaults to None.
        Return:
           img_feats (list[torch.Tensor]): List of features on every frame.
        """
        batch_size, num_frame, num_cam = img.shape[0], img.shape[1], img.shape[2]

        # backbone images
        # get all the images and let backbone infer for once
        all_imgs, all_img_metas = list(), list()
        for frame_idx in range(num_frame):
            # single frame image, N * NumCam * C * H * W
            img_single_frame = torch.stack([p[frame_idx] for p in img], dim=0)
            all_imgs.append(img_single_frame)
        # image metas
        img_metas_keys = img_metas[0].keys()
        for i in range(batch_size):
            img_metas_single_sample = dict()
            for key in img_metas_keys:
                # join the fields of every timestamp
                if type(img_metas[i][key][0]) == list:
                    contents = deepcopy(img_metas[i][key][0])
                    for j in range(1, num_frame):
                        contents += img_metas[i][key][j]
                # pick the representative one
                else:
                    contents = deepcopy(img_metas[i][key][0])
            img_metas_single_sample[key] = contents
            all_img_metas.append(img_metas_single_sample)

        # all imgs N * (T * NumCam) * C * H * W
        all_imgs = torch.cat(all_imgs, dim=1)
        # img_feats List[Tensor of batch 0, ...], each tensor BS * (T * NumCam) * C * H * W
        all_img_feats = self.extract_feat(img=all_imgs, img_metas=all_img_metas)

        # per frame feature maps
        img_feats = list()
        for i in range(num_frame):
            single_frame_feats = [lvl_feats[:, num_cam * i: num_cam * (i + 1), :, :, :] for lvl_feats in all_img_feats]
            img_feats.append(single_frame_feats)
        return img_feats
    
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        # print(img[0].size())
        if isinstance(img, list):
            img = torch.stack(img, dim=0)
        
        context = nullcontext()
        if not self.train_backbone:
            context = torch.no_grad()

        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)
            if img.dim() == 5:
                if img.size(0) == 1 and img.size(1) != 1:
                    img.squeeze_()
                else:
                    B, N, C, H, W = img.size()
                    img = img.view(B * N, C, H, W)
            with context:
                if self.use_grid_mask:
                    img = self.grid_mask(img)
                img_feats = self.img_backbone(img)
                if isinstance(img_feats, dict):
                    img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    # @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        return img_feats
    
    @force_fp32(apply_to=('img', 'points'))
    def forward(self, return_loss=True, tracking=False, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        elif tracking:
            return self.forward_track(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    def init_params_and_layers(self):
        """Generate the instances for tracking, especially the object queries
        """
        # query initialization for detection
        # reference points, mapping fourier encoding to embed_dims
        self.reference_points = nn.Embedding(self.num_query, 3)
        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims*3//2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)

        # embedding initialization for tracking
        if self.tracking:
            self.query_feat_embedding = nn.Embedding(self.num_query, self.embed_dims)
            nn.init.zeros_(self.query_feat_embedding.weight)
        
        # freeze backbone
        if not self.train_backbone:
            for param in self.img_backbone.parameters():
                param.requires_grad = False
        return


def track_bbox3d2result(bboxes, scores, labels, obj_idxes, track_scores, forecasting=None, attrs=None):
    """Convert detection results to a list of numpy arrays.
    Args:
        bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
        labels (torch.Tensor): Labels with shape of (n, ).
        scores (torch.Tensor): Scores with shape of (n, ).
        forecasting (torch.Tensor): Motion forecasting with shape of (n, T, 2)
        attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
            Defaults to None.
    Returns:
        dict[str, torch.Tensor]: Bounding box results in cpu mode.
            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
            - forecasting (torh.Tensor, optional): Motion forecasting
    """
    result_dict = dict(
        boxes_3d=bboxes.to('cpu'),
        scores_3d=scores.cpu(),
        labels_3d=labels.cpu(),
        track_scores=track_scores.cpu(),
        track_ids=obj_idxes.cpu(),
    )

    if attrs is not None:
        result_dict['attrs_3d'] = attrs.cpu()
    
    if forecasting is not None:
        result_dict['forecasting'] = forecasting.cpu()[..., :2]

    return result_dict