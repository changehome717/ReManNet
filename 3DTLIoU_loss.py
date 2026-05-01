import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES, build_assigner
from .kornia_focal import FocalLoss
from .utils import get_class_weight, weight_reduce_loss


@LOSSES.register_module()
class LaneLoss(nn.Module):
    def __init__(self,
                 focal_alpha=0.25,
                 focal_gamma=2.,
                 anchor_len=10,
                 gt_anchor_len=200,
                 anchor_steps=[],
                 weighted_ce=False,
                 use_sigmoid=False,
                 loss_weights=None,
                 anchor_assign=True,
                 assign_cfg=None):
        super(LaneLoss, self).__init__()

        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        self.anchor_len = anchor_len
        self.gt_anchor_len = gt_anchor_len

        self.anchor_steps = np.array(anchor_steps) - 1
        self.y_coordinate = np.array(anchor_steps, dtype=np.float32)

        self.weighted_ce = weighted_ce
        self.use_sigmoid = use_sigmoid
        self.loss_weights = loss_weights

        self.anchor_assign = anchor_assign
        self.fp16_enabled = False

        self.assigner = build_assigner(assign_cfg)

    def Spatial_tunnel_iou_loss(self,
                                x_pred,
                                z_pred,
                                x_gt,
                                z_gt,
                                vis_target,
                                e=1.5,
                                eps=1e-12):
        """
        Spatial tunnel IoU loss in the x-z plane.
    
        Args:
            x_pred: [N, T]
            z_pred: [N, T]
            x_gt: [N, T]
            z_gt: [N, T]
            vis_target: [N, T]
            e: half width of the tunnel interval.
            eps: numerical stability term.
    
        Returns:
            Scalar loss.
    
        Note:
            The intersection term is intentionally not clamped to be non-negative.
            When two tunnel intervals do not overlap, inter_right - inter_left becomes
            negative. In this design, the negative value is used to indicate that the
            predicted point has moved away from the GT tunnel, thereby providing a
            stronger penalty for distant predictions.
        """
    
        dx = x_pred - x_gt
        dz = z_pred - z_gt
    
        radius = torch.hypot(dx, dz)
    
        side_sign = torch.where(
            dx >= 0,
            torch.ones_like(dx),
            -torch.ones_like(dx)
        )
    
        x_rotated = x_gt + side_sign * radius
    
        pred_left = x_rotated - e
        pred_right = x_rotated + e
    
        gt_left = x_gt - e
        gt_right = x_gt + e
    
        inter_left = torch.maximum(pred_left, gt_left)
        inter_right = torch.minimum(pred_right, gt_right)
    
        union_left = torch.minimum(pred_left, gt_left)
        union_right = torch.maximum(pred_right, gt_right)
    
        # Do not clamp the intersection to zero.
        # A negative intersection means the two tunnel intervals are separated.
        # This negative value is intentionally preserved to penalize predictions
        # that move farther away from the GT tunnel.
        intersection = inter_right - inter_left
    
        union = union_right - union_left + eps
    
        visible_intersection = intersection * vis_target
        visible_union = union * vis_target
    
        lane_iou = visible_intersection.sum(dim=-1) / (
            visible_union.sum(dim=-1) + eps
        )
    
        loss = 1.0 - lane_iou
        return loss.mean()

    def vector_similarity_loss(self,
                               x_pred: torch.Tensor,
                               z_pred: torch.Tensor,
                               x_gt: torch.Tensor,
                               z_gt: torch.Tensor,
                               y_steps: torch.Tensor,
                               vis_target: torch.Tensor,
                               thr: float = 0.5,
                               eps: float = 1e-12):
        """
        Directional vector similarity loss for visible 3D lane segments.

        Args:
            x_pred: [N, T]
            z_pred: [N, T]
            x_gt: [N, T]
            z_gt: [N, T]
            y_steps: [T]
            vis_target: [N, T]
            thr: visibility threshold.
            eps: numerical stability term.

        Returns:
            Scalar loss.
        """

        num_lanes, num_points = x_pred.shape
        device = x_pred.device
        dtype = x_pred.dtype

        visible_mask = vis_target > thr

        point_order = torch.arange(
            num_points,
            device=device,
            dtype=dtype
        ).view(1, num_points).expand(num_lanes, num_points)

        sorting_key = (~visible_mask).to(dtype) * 1e6 + point_order
        sorting_index = torch.argsort(sorting_key, dim=1)

        def gather_visible_first(tensor):
            return torch.gather(tensor, dim=1, index=sorting_index)

        x_pred_sorted = gather_visible_first(x_pred)
        z_pred_sorted = gather_visible_first(z_pred)

        x_gt_sorted = gather_visible_first(x_gt)
        z_gt_sorted = gather_visible_first(z_gt)

        y_expand = y_steps.to(device=device, dtype=dtype)
        y_expand = y_expand.view(1, num_points).expand(num_lanes, num_points)
        y_sorted = torch.gather(y_expand, dim=1, index=sorting_index)

        mask_sorted = torch.gather(
            visible_mask,
            dim=1,
            index=sorting_index
        )

        visible_count = mask_sorted.sum(dim=1)

        column_index = torch.arange(
            num_points,
            device=device
        ).view(1, num_points).expand(num_lanes, num_points)

        compact_mask = column_index < visible_count.view(-1, 1)

        zero_value = x_pred_sorted.new_zeros(())

        x_pred_compact = torch.where(compact_mask, x_pred_sorted, zero_value)
        z_pred_compact = torch.where(compact_mask, z_pred_sorted, zero_value)

        x_gt_compact = torch.where(compact_mask, x_gt_sorted, zero_value)
        z_gt_compact = torch.where(compact_mask, z_gt_sorted, zero_value)

        y_compact = torch.where(compact_mask, y_sorted, zero_value)

        segment_mask = compact_mask[:, :-1] & compact_mask[:, 1:]

        pred_dx = x_pred_compact[:, 1:] - x_pred_compact[:, :-1]
        pred_dz = z_pred_compact[:, 1:] - z_pred_compact[:, :-1]

        gt_dx = x_gt_compact[:, 1:] - x_gt_compact[:, :-1]
        gt_dz = z_gt_compact[:, 1:] - z_gt_compact[:, :-1]

        dy = y_compact[:, 1:] - y_compact[:, :-1]

        dot_product = pred_dx * gt_dx + dy * dy + pred_dz * gt_dz

        pred_norm = torch.sqrt(
            pred_dx * pred_dx + dy * dy + pred_dz * pred_dz + eps
        )

        gt_norm = torch.sqrt(
            gt_dx * gt_dx + dy * dy + gt_dz * gt_dz + eps
        )

        cosine = dot_product / (pred_norm * gt_norm + eps)
        cosine = torch.clamp(cosine, min=-1.0, max=1.0)

        similarity = (cosine + 1.0) * 0.5

        segment_mask_float = segment_mask.to(similarity.dtype)

        similarity_sum = (similarity * segment_mask_float).sum(dim=1)
        valid_segment_count = segment_mask_float.sum(dim=1).clamp_min(1.0)

        lane_similarity = similarity_sum / valid_segment_count

        loss = 1.0 - lane_similarity
        return loss.mean()

    def forward(self, proposals_list, targets):
        focal_loss_fn = FocalLoss(
            alpha=self.focal_alpha,
            gamma=self.focal_gamma
        )

        smooth_l1_fn = nn.SmoothL1Loss(reduction='none')

        device = proposals_list[0][0].device
        batch_size = len(proposals_list)

        loss_accumulator = {
            'cls_loss': torch.zeros((), device=device),
            'reg_losses_x': torch.zeros((), device=device),
            'reg_losses_z': torch.zeros((), device=device),
            'reg_losses_vis': torch.zeros((), device=device),
            'ST_iou_loss': torch.zeros((), device=device),
            'cos_similarity_loss': torch.zeros((), device=device),
        }

        total_positives = 0
        total_negatives = 0

        def cls_start_index():
            return 5 + self.anchor_len * 3

        def get_classification_logits(proposals):
            return proposals[:, cls_start_index():]

        def add_zero_regression_terms(reference_tensor):
            zero_loss = reference_tensor.sum() * 0.0
            loss_accumulator['reg_losses_x'] += zero_loss
            loss_accumulator['reg_losses_z'] += zero_loss
            loss_accumulator['reg_losses_vis'] += zero_loss
            loss_accumulator['ST_iou_loss'] += zero_loss
            loss_accumulator['cos_similarity_loss'] += zero_loss

        def add_background_classification_loss(proposals):
            cls_logits = get_classification_logits(proposals)

            cls_target = torch.zeros(
                cls_logits.size(0),
                dtype=torch.long,
                device=cls_logits.device
            )

            cls_loss = focal_loss_fn(cls_logits, cls_target)
            loss_accumulator['cls_loss'] += cls_loss.sum()

            add_zero_regression_terms(cls_logits)

        def sample_dense_target(raw_target):
            """
            Convert the dense target lane representation into the sampled
            anchor format used by the prediction head.

            Output layout:
            [meta(5), x(anchor_len), z(anchor_len), visibility(anchor_len)]
            """

            sample_indices = torch.as_tensor(
                self.anchor_steps,
                dtype=torch.long,
                device=raw_target.device
            ) + 5

            x_indices = sample_indices
            z_indices = sample_indices + self.gt_anchor_len
            vis_indices = sample_indices + self.gt_anchor_len * 2

            sampled_x = raw_target.index_select(1, x_indices)
            sampled_z = raw_target.index_select(1, z_indices)
            sampled_vis = raw_target.index_select(1, vis_indices)

            sampled_target = torch.cat(
                [
                    raw_target[:, :5],
                    sampled_x,
                    sampled_z,
                    sampled_vis
                ],
                dim=1
            )

            return sampled_target

        for (proposal_pack, target) in zip(proposals_list, targets):
            proposals, anchors = proposal_pack

            num_classes = proposals.shape[1] - 5 - self.anchor_len * 3

            valid_target = target[target[:, 1] > 0]

            if valid_target.numel() == 0:
                add_background_classification_loss(proposals)
                continue

            sampled_target = sample_dense_target(valid_target)

            with torch.no_grad():
                if self.anchor_assign:
                    assignment_source = anchors
                else:
                    assignment_source = proposals[:, :5 + self.anchor_len * 3]

                positives_mask, negatives_mask, matched_target_indices = \
                    self.assigner.match_proposals_with_targets(
                        assignment_source,
                        sampled_target
                    )

            positive_proposals = proposals[positives_mask]
            negative_proposals = proposals[negatives_mask]

            num_positives = positive_proposals.size(0)
            num_negatives = negative_proposals.size(0)

            total_positives += num_positives
            total_negatives += num_negatives

            if num_positives == 0:
                add_background_classification_loss(proposals)
                continue

            matched_target = sampled_target[matched_target_indices]

            all_sampled_proposals = torch.cat(
                [positive_proposals, negative_proposals],
                dim=0
            )

            cls_logits = get_classification_logits(all_sampled_proposals)

            cls_target = torch.zeros(
                cls_logits.size(0),
                dtype=torch.long,
                device=cls_logits.device
            )

            cls_target[:num_positives] = matched_target[:, 1].long()

            cls_loss = focal_loss_fn(cls_logits, cls_target)

            if self.use_sigmoid:
                cls_normalizer = max(num_positives * num_classes, 1)
            else:
                cls_normalizer = max(num_positives, 1)

            loss_accumulator['cls_loss'] += cls_loss.sum() / cls_normalizer

            x_pred = positive_proposals[:, 5:5 + self.anchor_len]

            z_pred_start = 5 + self.anchor_len
            z_pred_end = 5 + self.anchor_len * 2
            z_pred = positive_proposals[:, z_pred_start:z_pred_end]

            vis_pred_start = 5 + self.anchor_len * 2
            vis_pred_end = 5 + self.anchor_len * 3
            vis_pred = positive_proposals[:, vis_pred_start:vis_pred_end]

            with torch.no_grad():
                x_gt = matched_target[:, 5:5 + self.anchor_len]

                z_gt_start = 5 + self.anchor_len
                z_gt_end = 5 + self.anchor_len * 2
                z_gt = matched_target[:, z_gt_start:z_gt_end]

                vis_gt_start = 5 + self.anchor_len * 2
                vis_gt_end = 5 + self.anchor_len * 3
                vis_gt = matched_target[:, vis_gt_start:vis_gt_end]

                valid_points = vis_gt.sum().clamp_min(1.0)

            x_reg_loss = smooth_l1_fn(x_pred, x_gt) * vis_gt
            z_reg_loss = smooth_l1_fn(z_pred, z_gt) * vis_gt
            vis_reg_loss = smooth_l1_fn(vis_pred, vis_gt)

            loss_accumulator['reg_losses_x'] += x_reg_loss.sum() / valid_points
            loss_accumulator['reg_losses_z'] += z_reg_loss.sum() / valid_points
            loss_accumulator['reg_losses_vis'] += vis_reg_loss.mean()

            st_iou_loss = self.Spatial_tunnel_iou_loss(
                x_pred=x_pred,
                z_pred=z_pred,
                x_gt=x_gt,
                z_gt=z_gt,
                vis_target=vis_gt
            )

            loss_accumulator['ST_iou_loss'] += st_iou_loss

            y_steps = torch.as_tensor(
                self.y_coordinate,
                dtype=x_pred.dtype,
                device=x_pred.device
            )

            cos_loss = self.vector_similarity_loss(
                x_pred=x_pred,
                z_pred=z_pred,
                x_gt=x_gt,
                z_gt=z_gt,
                y_steps=y_steps,
                vis_target=vis_gt
            )

            loss_accumulator['cos_similarity_loss'] += cos_loss

        losses = {
            name: value / batch_size
            for name, value in loss_accumulator.items()
        }

        if self.loss_weights is not None:
            for name in losses:
                if name in self.loss_weights:
                    losses[name] = losses[name] * self.loss_weights[name]

        return {
            'losses': losses,
            'batch_positives': total_positives / batch_size,
            'batch_negatives': total_negatives / batch_size
        }
