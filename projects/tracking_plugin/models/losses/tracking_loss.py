# ------------------------------------------------------------------------
# Copyright (c) Toyota Research Institute
# ------------------------------------------------------------------------
# Modified from PETR (https://github.com/megvii-research/PETR)
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet.models import LOSSES
from mmdet.models import build_loss
from mmdet.core import (build_assigner, reduce_mean, multi_apply, build_sampler)
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from .tracking_loss_base import TrackingLossBase


@LOSSES.register_module()
class TrackingLoss(TrackingLossBase):
    def __init__(self,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)
    
    def loss_single_frame(self,
                          frame_idx,
                          gt_bboxes_list,
                          gt_labels_list,
                          instance_ids,
                          preds_dicts,
                          gt_bboxes_ignore):
        """Match according to both tracking and detection information
           Generate the single frame loss function, modify the ids of track instances
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        
        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        track_instances = preds_dicts['track_instances']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        # after this operation, [x, y, z-h/2] becomes [x, y, z]
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]
        
        obj_idxes_list = instance_ids[0].detach().cpu().numpy().tolist()
        obj_idx_to_gt_idx = {obj_idx: gt_idx for gt_idx, obj_idx in enumerate(obj_idxes_list)}

        num_disappear_track = 0
        # step 1. Inherit and Update the previous tracks
        for trk_idx in range(len(track_instances)):
            obj_id = track_instances.obj_idxes[trk_idx].item()
            if obj_id >= 0:
                if obj_id in obj_idx_to_gt_idx:
                    track_instances.matched_gt_idxes[trk_idx] = obj_idx_to_gt_idx[obj_id]
                else:
                    num_disappear_track += 1
                    track_instances.matched_gt_idxes[trk_idx] = -2
            else:
                track_instances.matched_gt_idxes[trk_idx] = -1
        
        full_track_idxes = torch.arange(len(track_instances), dtype=torch.long).to(all_cls_scores.device)
        # previsouly tracked, which is matched by rule
        all_matched_track_idxes = full_track_idxes[track_instances.obj_idxes >= 0]
        matched_track_idxes = full_track_idxes[track_instances.matched_gt_idxes >= 0]
        
        # step2. select the unmatched slots.
        # note that the FP tracks whose obj_idxes are -2 will not be selected here.
        unmatched_track_idxes = full_track_idxes[track_instances.obj_idxes == -1]

        # step3. select the untracked gt instances (new tracks).
        tgt_indexes = track_instances.matched_gt_idxes.clone()
        tgt_indexes = tgt_indexes[tgt_indexes >= 0]
        tgt_state = torch.zeros(len(gt_bboxes_list[0])).to(all_cls_scores.device)
        tgt_state[tgt_indexes] = 1
        # new tgt indexes
        untracked_tgt_indexes = torch.arange(len(gt_bboxes_list[0])).to(all_cls_scores.device)[tgt_state == 0]

        all_unmatched_gt_bboxes_list = [[gt_bboxes_list[0][untracked_tgt_indexes]] for _ in range(num_dec_layers)]
        all_unmatched_gt_labels_list = [[gt_labels_list[0][untracked_tgt_indexes]] for _ in range(num_dec_layers)]
        all_unmatched_gt_ids_list = [[instance_ids[0][untracked_tgt_indexes]] for _ in range(num_dec_layers)]
        all_unmatched_ignore_list = [None for _ in range(num_dec_layers)]

        unmatched_cls_scores = all_cls_scores[:, :, unmatched_track_idxes, :]
        unmatched_bbox_preds = all_bbox_preds[:, :, unmatched_track_idxes, :]

        # step4. do matching between the unmatched slots and GTs.
        unmatched_track_matching_result = list()
        for dec_layer_idx in range(num_dec_layers):
            unmatched_track_dec_matching_result = self.get_targets(
                unmatched_cls_scores[dec_layer_idx],
                unmatched_bbox_preds[dec_layer_idx],
                all_unmatched_gt_bboxes_list[dec_layer_idx],
                all_unmatched_gt_labels_list[dec_layer_idx],
                all_unmatched_gt_ids_list[dec_layer_idx],
                all_unmatched_ignore_list[dec_layer_idx])
            
            unmatched_track_matching_result.append(unmatched_track_dec_matching_result)
            if dec_layer_idx == num_dec_layers - 1:
                (labels_list, label_instance_ids_list, label_weights_list, bbox_targets_list,
                    bbox_weights_list, num_total_pos, num_total_neg, gt_match_idxes_list) = unmatched_track_dec_matching_result
        
        # step5. update the obj_idxes according to the matching result with the last decoder layer
        track_instances.obj_idxes[unmatched_track_idxes] = label_instance_ids_list[0]
        track_instances.matched_gt_idxes[unmatched_track_idxes] = gt_match_idxes_list[0]

        # step6. merge the matching results of tracking/query instances
        matched_labels = gt_labels_list[0][tgt_indexes].long()
        matched_label_weights = gt_labels_list[0].new_ones(len(tgt_indexes)).float()
        matched_bbox_targets = gt_bboxes_list[0][tgt_indexes]
        matched_bbox_weights = torch.ones_like(track_instances.bboxes)[:len(tgt_indexes)]

        all_matching_list = list()
        matched_track_idxes = full_track_idxes[matched_track_idxes]
        unmatched_track_idxes = full_track_idxes[unmatched_track_idxes]

        for dec_layer_idx in range(num_dec_layers):
            (dec_labels, _, dec_label_weights, dec_bbox_targets,
                dec_bbox_weights, dec_num_total_pos, dec_num_total_neg, _) = unmatched_track_matching_result[dec_layer_idx]

            labels_list = torch.ones_like(track_instances.obj_idxes).long() * self.num_classes
            labels_list[matched_track_idxes] = matched_labels
            labels_list[unmatched_track_idxes] = dec_labels[0]
            labels_list = [labels_list]

            label_weights_list = torch.ones_like(track_instances.obj_idxes).float()
            label_weights_list = [label_weights_list]

            bbox_targets_list = torch.zeros_like(track_instances.bboxes)[:, :dec_bbox_targets[0].size(1)]
            bbox_targets_list[matched_track_idxes] = matched_bbox_targets
            bbox_targets_list[unmatched_track_idxes] = dec_bbox_targets[0]
            bbox_targets_list = [bbox_targets_list]

            bbox_weights_list = torch.zeros_like(track_instances.bboxes)
            bbox_weights_list[matched_track_idxes] = 1.0
            bbox_weights_list[unmatched_track_idxes] = dec_bbox_weights[0]
            bbox_weights_list = [bbox_weights_list]
            
            total_pos = dec_num_total_pos + len(matched_track_idxes)
            total_neg = dec_num_total_neg + num_disappear_track

            matched_gt_idxes_list = track_instances.obj_idxes.new_full((len(track_instances),), -1, dtype=torch.long)
            matched_gt_idxes_list[matched_track_idxes] = track_instances.matched_gt_idxes[matched_track_idxes]
            matched_gt_idxes_list[unmatched_track_idxes] = track_instances.matched_gt_idxes[unmatched_track_idxes]

            dec_matching_results = (labels_list, label_weights_list, bbox_targets_list,
                                    bbox_weights_list, total_pos, total_neg, matched_gt_idxes_list)
            all_matching_list.append(dec_matching_results)
        
        # step 7. compute the single frame losses
        # after getting the matching result, we no longer need contents for gt_bboxes_list etc.
        if self.interm_loss:
            losses_cls, losses_bbox = multi_apply(
               self.loss_single_decoder, [frame_idx for _ in range(num_dec_layers)], 
               all_cls_scores, all_bbox_preds,
               [None for _ in range(num_dec_layers)], [None for _ in range(num_dec_layers)], 
               [None for _ in range(num_dec_layers)], [None for _ in range(num_dec_layers)], 
               all_matching_list)
        else:
            losses_cls, losses_bbox = self.loss_single_decoder(frame_idx,
                all_cls_scores[-1], all_bbox_preds[-1],
                None, None, None, None, all_matching_list[-1])
            losses_cls, losses_bbox = [losses_cls], [losses_bbox]
        
        loss_dict = dict()

        # loss from the last decoder layer
        loss_dict[f'f{frame_idx}.loss_cls'] = losses_cls[-1]
        loss_dict[f'f{frame_idx}.loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'f{frame_idx}.d{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'f{frame_idx}.d{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1

        return loss_dict
