import sys

import torch
from torch.nn import functional as F
from typing import List, Optional


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        valid = None,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    if valid is not None:
        inputs,targets = inputs,targets
    numerator = 2 * (inputs * targets).sum(-1).sum(-1)
    denominator = inputs.sum(-1).sum(-1) + targets.sum(-1).sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    loss = loss.sum() 

    return loss


def mean_square_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor
):
    loss = F.mse_loss(inputs, targets)
    return loss





def loss_iou_mse(pred_iou, src_masks, target_masks):
    """
    pred_iou.shape: (bs, 1)

    (2) get mse loss with pred_iou & gt_iou
    """
    with torch.no_grad():
        src_masks = src_masks.type(torch.int).flatten(1)  # (bs,1024*1024)
        target_masks = (target_masks > 0.5).type(torch.int).flatten(1)
        area_src = src_masks.sum(dim=1)
        area_target = target_masks.sum(dim=1)
        intersection = (src_masks * target_masks).sum(dim=1)
        gt_iou = intersection / (area_src + area_target - intersection + 1e-6)
        gt_iou = gt_iou.reshape(-1, 1)
    loss_mse = mean_square_loss(pred_iou, gt_iou)
    del src_masks
    del target_masks
    return loss_mse


def loss_hi_iou_mse(pred_iou, src_masks, mask_thresh, target_masks):
    # src_masks: logits
    with torch.no_grad():
        target_masks = F.interpolate(target_masks, src_masks.shape[-2:], mode="bilinear", align_corners=False)
        target_masks = (target_masks > 0.5).type(torch.int).flatten(1)
        src_masks = (src_masks > mask_thresh).type(torch.int).flatten(1)
        area_src = src_masks.sum(dim=1)
        area_target = target_masks.sum(dim=1)
        intersection = (src_masks * target_masks).sum(dim=1)
        gt_iou = intersection / (area_src + area_target - intersection + 1e-6)
        gt_iou = gt_iou.reshape(-1, 1)
    loss_mse = mean_square_loss_jit(pred_iou, gt_iou)
    del src_masks
    del target_masks
    return loss_mse