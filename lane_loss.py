# --------------------------------------------------------
# Source code for Anchor3DLane
# Copyright (c) 2023 TuSimple
# @Time    : 2023/04/05
# @Author  : Shaofei Huang
# nowherespyfly@gmail.com
# --------------------------------------------------------

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
        self.anchor_steps = np.array(anchor_steps) - 1
        self.gt_anchor_len = gt_anchor_len
        self.use_sigmoid = use_sigmoid
        self.y_coordinate = np.array(anchor_steps, dtype=np.float32)

        self.weighted_ce = weighted_ce
        self.loss_weights = loss_weights
        self.anchor_assign = anchor_assign
        self.fp16_enabled = False
        self.assigner = build_assigner(assign_cfg)
        
        
    def Spatial_tunnel_iou_loss(self, x_pred, z_pred, x_gt, z_gt, vis_target, e=1.5, eps=1e-12):
        """
        x_pred, z_pred, x_gt, z_gt: [a, 20]
        e: 半宽（默认 1.5）
        返回:
            iou_pts:   [a, 20]  每个对应点的 IoU
            iou_lane:  [a]      每条车道(20点平均)的 IoU
            iou_mean:  ()       全部点的均值 IoU
        """
        # 旋转：将 P(xp,zp) 绕 G(xg,zg) 旋到 z'=zg（在圆上选与原 dx 同侧的交点）
        dx = x_pred - x_gt          # [a,20]
        dz = z_pred - z_gt          # [a,20]
        r  = torch.hypot(dx, dz)    # 半径: sqrt(dx^2 + dz^2)

        sign = torch.where(dx >= 0, torch.ones_like(dx), -torch.ones_like(dx))
        x_prime = x_gt + sign * r   # 旋转后的 x_p'
        # z_prime = z_gt            # 若需要可返回，这里 IoU 只用 x

        # 区间 IoU： [x'-e, x'+e] 与 [xg-e, xg+e]
        a1, a2 = x_prime - e, x_prime + e
        b1, b2 = x_gt    - e, x_gt    + e

        inter = torch.minimum(a2, b2) - torch.maximum(a1, b1)
        union = torch.maximum(a2, b2) - torch.minimum(a1, b1)
        iou_lane = (inter * vis_target).sum(dim=-1) /((union * vis_target).sum(dim=-1) + eps)   # [a,20]

        # 汇总
        iou_lane_mean = (1-iou_lane).mean()                    
        return torch.tensor(iou_lane_mean, device=x_gt.device)

    


    def vector_similarity_loss(self,
        x_pred: torch.Tensor,  # [a, T]
        z_pred: torch.Tensor,  # [a, T]
        x_gt:   torch.Tensor,  # [a, T]
        z_gt:   torch.Tensor,  # [a, T]
        y_steps: torch.Tensor, # [T]
        vis_target: torch.Tensor,  # [a, T] (0/1 或概率)
        thr: float = 0.5,
        eps: float = 1e-12):
        """
        1) 用 vis_target 选出可见点，并把每条车道的可见点“挪到前面”，尾部用 0 填充到 T。
        2) 只对相邻两点均有效的段计算 3D 余弦相似度（x,y,z）。
        返回标量损失：mean(1 - cos_seg)，仅对有效段聚合。
        """
        a, T = x_pred.shape
        device, dtype = x_pred.device, x_pred.dtype

        # --- 可见点掩码 & 索引排序：可见在前、不可见在后（保持相对时序稳定） ---
        mask = (vis_target > thr)                            # [a, T]
        base_idx = torch.arange(T, device=device).view(1, T).expand(a, T).to(dtype)
        # 可见=0，不可见=1e6；同一分组内按原序（base_idx）稳定排序
        sort_key = (~mask).to(dtype) * 1e6 + base_idx        # [a, T]
        sort_idx = torch.argsort(sort_key, dim=1)  # [a, T]

        # 重新排列（“可见在前”）
        def _gather(x):
            return torch.gather(x, 1, sort_idx)
        x_pred_s = _gather(x_pred)    # [a, T]
        z_pred_s = _gather(z_pred)
        x_gt_s   = _gather(x_gt)
        z_gt_s   = _gather(z_gt)
        y_s      = torch.gather(y_steps.to(device=device, dtype=dtype).view(1, T).expand(a, T), 1, sort_idx)
        m_s      = torch.gather(mask, 1, sort_idx)           # [a, T]

        # 每条车道有效点数
        counts = m_s.sum(dim=1)                              # [a]

        # 构造前缀有效掩码（用于后续段筛选）
        col = torch.arange(T, device=device).view(1, T).expand(a, T)
        head_mask = col < counts.view(-1, 1)                 # [a, T], 前 counts[i] 为 True

        # --- 尾部填充 0（只作为占位；真正是否参与计算由 head_mask 决定） ---
        pad0 = x_pred_s.new_zeros(())
        x_pred_c = torch.where(head_mask, x_pred_s, pad0)
        z_pred_c = torch.where(head_mask, z_pred_s, pad0)
        x_gt_c   = torch.where(head_mask, x_gt_s,   pad0)
        z_gt_c   = torch.where(head_mask, z_gt_s,   pad0)
        y_c      = torch.where(head_mask, y_s,      pad0)

        # --- 构造有效段掩码：相邻两点都有效才算一段 ---
        seg_mask = head_mask[:, :-1] & head_mask[:, 1:]      # [a, T-1]

        # 相邻点差分 -> 段向量
        dx_p = x_pred_c[:, 1:] - x_pred_c[:, :-1]            # [a, T-1]
        dz_p = z_pred_c[:, 1:] - z_pred_c[:, :-1]
        dx_g = x_gt_c[:, 1:]   - x_gt_c[:, :-1]
        dz_g = z_gt_c[:, 1:]   - z_gt_c[:, :-1]
        dy   = y_c[:, 1:]      - y_c[:, :-1]                 # 注意：删除点后，相邻 y 已按可见序对齐

        # 3D 余弦相似度（仅在有效段上统计）
        dot   = dx_p * dx_g + dy * dy + dz_p * dz_g          # [a, T-1]
        norm_p = torch.sqrt(dx_p * dx_p + dy * dy + dz_p * dz_p + eps)
        norm_g = torch.sqrt(dx_g * dx_g + dy * dy + dz_g * dz_g + eps)
        cos_seg = (torch.clamp(dot / (norm_p * norm_g + eps), -1.0, 1.0) + 1) * 0.5

        # 用 seg_mask 做“掩码平均”：只聚合有效段
        seg_mask_f = seg_mask.to(cos_seg.dtype)
        cos_sum_per_lane   = (cos_seg * seg_mask_f).sum(dim=1)                 # [a]
        valid_segs_per_lane = seg_mask_f.sum(dim=1).clamp_min(1.0)             # [a]
        cos_lane = cos_sum_per_lane / valid_segs_per_lane                      # [a]

        # 标量损失：mean(1 - cos_lane)
        cos_simi_loss = (1.0 - cos_lane).mean()
        return torch.tensor(cos_simi_loss, device=x_gt.device)


        
    def forward(self, proposals_list, targets):
        focal_loss = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)
        smooth_l1_loss = nn.SmoothL1Loss(reduction='none')
        cls_losses = 0
        reg_losses_x = 0
        reg_losses_z = 0
        reg_losses_vis = 0
        ST_iou_loss = torch.tensor(0.0, device=proposals_list[0][0].device)
        cos_similarity_loss = torch.tensor(0.0, device=proposals_list[0][0].device)
        valid_imgs = len(targets)
        total_positives = 0
        total_negatives = 0
        for idx, ((proposals, anchors), target) in enumerate(zip(proposals_list, targets)):
            # Filter lanes that do not exist (confidence == 0)
            num_clses = proposals.shape[1] - 5 - self.anchor_len * 3
            target = target[target[:, 1] > 0]   # [N, 605]
            if len(target) == 0:
                # If there are no targets, all proposals have to be negatives (i.e., 0 confidence)
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, 5+self.anchor_len*3:]
                cls_losses += focal_loss(cls_pred, cls_target).sum()
                reg_losses_x += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                reg_losses_z += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                reg_losses_vis += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                continue
            # Gradients are also not necessary for the positive & negative matching
            x_indices = torch.tensor(self.anchor_steps).to(torch.long).to(target.device) + 5
            z_indices = x_indices + self.gt_anchor_len
            vis_indices = x_indices + self.gt_anchor_len * 2
            x_target = target.index_select(1, x_indices)
            z_target = target.index_select(1, z_indices)
            vis_target = target.index_select(1, vis_indices)   # [N, 10]
            target = torch.cat((target[:, :5], x_target, z_target, vis_target), dim=1)   # [N, 35]
            with torch.no_grad():
                if self.anchor_assign:
                    positives_mask, negatives_mask, target_positives_indices = self.assigner.match_proposals_with_targets(
                        anchors, target)
                else:
                    positives_mask, negatives_mask, target_positives_indices = self.assigner.match_proposals_with_targets(
                        proposals[:, :5+self.anchor_len*3], target)

            positives = proposals[positives_mask]
            num_positives = len(positives)
            total_positives += num_positives
            negatives = proposals[negatives_mask]
            num_negatives = len(negatives)
            total_negatives += num_negatives

            # Handle edge case of no positives found
            if num_positives == 0:
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, :2]
                cls_losses += focal_loss(cls_pred, cls_target).sum()
                reg_losses_x += smooth_l1_loss(cls_pred, cls_pred).sum() * 0  # avoid dividing zeros
                reg_losses_z += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                reg_losses_vis += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                continue

            # Get classification targets
            all_proposals = torch.cat([positives, negatives], 0)
            cls_target = proposals.new_zeros(num_positives + num_negatives).long()
            cls_target[:num_positives] = target[target_positives_indices][:, 1]
            cls_pred = all_proposals[:, 5+self.anchor_len*3:]  # [N, C]

            # Regression targets
            x_pred = positives[:, 5:5+self.anchor_len]   # [N, l]
            z_pred = positives[:, 5+self.anchor_len:5+self.anchor_len*2]   # [N, l]
            
            vis_pred = positives[:, 5+self.anchor_len*2:5+self.anchor_len*3]  # [N, l]
            with torch.no_grad():
                target = target[target_positives_indices]
                x_target = target[:, 5:5+self.anchor_len]
                z_target = target[:, 5+self.anchor_len:5+self.anchor_len*2]
                vis_target = target[:, 5+self.anchor_len*2:5+self.anchor_len*3]
                valid_points = vis_target.sum()

            # Loss calc
            reg_loss_x = smooth_l1_loss(x_pred, x_target)
            reg_loss_x = reg_loss_x * vis_target  #  * scores # [N, l]
            reg_losses_x += reg_loss_x.sum() / valid_points
            reg_loss_z = smooth_l1_loss(z_pred, z_target)
            reg_loss_z = reg_loss_z * vis_target # * scores
            reg_losses_z += reg_loss_z.sum() / valid_points
            reg_loss_vis = smooth_l1_loss(vis_pred, vis_target)
            reg_losses_vis += reg_loss_vis.mean()
            cls_loss = focal_loss(cls_pred, cls_target)
        
            ST_iou_loss += self.Spatial_tunnel_iou_loss(x_pred, z_pred, x_target, z_target, vis_target)
            
            y_coordinate = torch.tensor(self.y_coordinate, device=x_pred.device, dtype=x_pred.dtype)
            cos_similarity_loss += self.vector_similarity_loss(x_pred, z_pred, x_target, z_target, y_coordinate, vis_target)
            
            if self.use_sigmoid:
                cls_losses += cls_loss.sum() / num_positives / num_clses
            else:
                cls_losses += cls_loss.sum() / num_positives

        # Batch mean
        cls_losses = cls_losses / valid_imgs
        reg_losses_x = reg_losses_x / valid_imgs
        reg_losses_z = reg_losses_z / valid_imgs
        reg_losses_vis = reg_losses_vis / valid_imgs
        ST_iou_loss = ST_iou_loss / valid_imgs
        cos_similarity_loss = cos_similarity_loss / valid_imgs

        losses = {'cls_loss': cls_losses, 'reg_losses_x': reg_losses_x, 'reg_losses_z': reg_losses_z, 'reg_losses_vis': reg_losses_vis, 'ST_iou_loss': ST_iou_loss, 'cos_similarity_loss': cos_similarity_loss}

        for k in losses.keys():
            losses[k] = losses[k] * self.loss_weights[k]

        bs = len(proposals_list)
        return {'losses':losses, 'batch_positives': total_positives / bs, 'batch_negatives': total_negatives / bs}