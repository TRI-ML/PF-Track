# ------------------------------------------------------------------------
# Copyright (c) 2023 toyota research instutute.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import torch
from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
import torch.nn.functional as F


@BBOX_CODERS.register_module()
class TrackNMSFreeCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detector. Including the fields for tracking
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10):
        
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds, obj_idxes=None, track_scores=None, motion_forecasting=None, masks=None):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
            obj_idxes (Tensor): The idxes of the track instances
            track_scores (Tensor): The scores of the bbox
            motion_forecasting (Tensor): The predicted trajectories, [num_query, T, 2]
            all_masks (Tensor): The masks for valid query output
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num
        cls_scores = cls_scores.sigmoid()

        if masks is not None:
            cls_scores = cls_scores[masks]
            bbox_preds = bbox_preds[masks]
            obj_idxes = obj_idxes[masks]
            track_scores = track_scores[masks]
            if motion_forecasting is not None:
                motion_forecasting = motion_forecasting[masks]
        
        # tracking mode decode
        if obj_idxes is not None:
            _, indexs = cls_scores.max(dim=-1)
            labels = indexs % self.num_classes
            _, bbox_index = track_scores.topk(min(max_num, len(obj_idxes)))
            track_scores = track_scores[bbox_index]
            obj_idxes = obj_idxes[bbox_index]
            bbox_preds = bbox_preds[bbox_index]
            labels = labels[bbox_index]
            scores = track_scores
            if motion_forecasting is not None:
                motion_forecasting = motion_forecasting[bbox_index]
        # detection mode decode
        else:
            cls_scores_topk = cls_scores.view(-1)
            scores, indexs = cls_scores_topk.topk(min(max_num, cls_scores_topk.size(0)))
            labels = indexs % self.num_classes
            scores, indexs = cls_scores_topk.topk(min(max_num, cls_scores_topk.size(0)))
            labels = indexs % self.num_classes
            bbox_index = indexs // self.num_classes
            bbox_preds = bbox_preds[bbox_index]

        final_scores = scores
        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)   
        final_preds = labels
        final_motion_forecasting = motion_forecasting

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(self.post_center_range, device=scores.device)
            
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            if final_motion_forecasting is not None:
                motion_forecasting = final_motion_forecasting[mask]
            if obj_idxes is not None:
                track_scores = track_scores[mask]
                obj_idxes = obj_idxes[mask]
            
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels,
                'track_scores': track_scores,
                'obj_idxes': obj_idxes,
                'forecasting': motion_forecasting
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
            track_instances (Instances): Instances containing track information. 
                Available for tracking evaluation.
        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['all_cls_scores'][-1].clone()
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1].clone()
        
        batch_size = all_cls_scores.size()[0]
        if 'track_instances' in preds_dicts.keys():
            track_instances = preds_dicts['track_instances'].clone()
            obj_idxes = [track_instances.obj_idxes.clone()]
            track_scores = [track_instances.scores.clone()]
            if 'all_masks' in preds_dicts.keys():
                all_masks = [preds_dicts['all_masks'].clone()]
            else:
                all_masks = [None]

            if 'all_motion_forecasting' in preds_dicts.keys() and preds_dicts['all_motion_forecasting'] is not None:
                motion_forecasting = preds_dicts['all_motion_forecasting'].clone()
                motion_forecasting = [motion_forecasting]
            else:
                motion_forecasting = [None]
        else:
            obj_idxes = [None for _ in range(batch_size)]
            track_scores = [None for _ in range(batch_size)]
            motion_forecasting = [None for _ in range(batch_size)]
            all_masks = [None for _ in range(batch_size)]

        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(
                all_cls_scores[i], all_bbox_preds[i], obj_idxes[i], track_scores[i], 
                motion_forecasting[i], all_masks[i]))
        return predictions_list