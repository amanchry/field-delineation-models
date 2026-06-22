from typing import Dict

import numpy as np
import torch
from scipy.ndimage import label as cc_label


def pixel_iou(pred: torch.Tensor, gt: torch.Tensor, num_classes: int = 3) -> Dict:
    """Per-class pixel IoU. Both tensors [H, W] with values 0..num_classes-1."""
    p  = pred.cpu().numpy()
    g  = gt.cpu().numpy()
    ious = {}
    for c in range(num_classes):
        inter = np.logical_and(p == c, g == c).sum()
        union = np.logical_or(p == c,  g == c).sum()
        ious[c] = float(inter) / float(union + 1e-6)
    ious["mean"] = float(np.mean(list(ious.values())))
    return ious


def object_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray,
                   iou_threshold: float = 0.5,
                   use_pred_instances: bool = False) -> Dict:
    """
    Match predicted field instances to GT instances by IoU >= iou_threshold.
    pred_mask : [H, W] — either semantic (1=field) or pre-labelled instance map
                         (integer per instance). use_pred_instances=True for the latter.
    gt_mask   : [H, W] semantic mask (1 = field pixel).
    """
    if use_pred_instances:
        pred_inst = pred_mask.astype(np.int32)
        n_pred    = int(pred_inst.max())
    else:
        pred_inst, n_pred = cc_label(pred_mask == 1)
    gt_inst,   n_gt   = cc_label(gt_mask   == 1)

    matched_pred, matched_gt = set(), set()
    for pid in range(1, n_pred + 1):
        p_bin = pred_inst == pid
        for gid in range(1, n_gt + 1):
            if gid in matched_gt:
                continue
            g_bin = gt_inst == gid
            inter = np.logical_and(p_bin, g_bin).sum()
            union = np.logical_or(p_bin,  g_bin).sum()
            if union > 0 and inter / union >= iou_threshold:
                matched_pred.add(pid)
                matched_gt.add(gid)
                break

    tp        = len(matched_pred)
    precision = tp / (n_pred + 1e-6)
    recall    = tp / (n_gt   + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "n_pred": n_pred, "n_gt": n_gt}
