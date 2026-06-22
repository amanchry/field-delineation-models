import torch
import torch.nn.functional as F


def dice_loss(pred_prob: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """Binary soft Dice loss. pred_prob and target must be same shape."""
    inter = (pred_prob * target).sum(dim=(-2, -1))
    denom = pred_prob.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return (1 - (2 * inter + smooth) / (denom + smooth)).mean()


def focal_loss(pred_prob: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: float = 0.5) -> torch.Tensor:
    """Binary focal loss."""
    fl = (
        -target * alpha * (1 - pred_prob) ** gamma * torch.log(pred_prob + 1e-5)
        - (1 - target) * (1 - alpha) * pred_prob ** gamma * torch.log(1 - pred_prob + 1e-5)
    )
    return fl.mean()


def sam2_loss(
    pred_logits: torch.Tensor,   # [N, 3, H, W]  raw logits
    pred_scores: torch.Tensor,   # [N, 3]         predicted IoU scores
    gt_masks: torch.Tensor,      # [N, H, W]      binary GT
) -> dict:
    """Focal + Dice loss on best-of-3 mask predictions, plus IoU score regression."""
    pred_prob = torch.sigmoid(pred_logits)
    ph, pw = pred_prob.shape[-2], pred_prob.shape[-1]
    gt = gt_masks.unsqueeze(1).float()
    if gt.shape[-2] != ph or gt.shape[-1] != pw:
        gt = F.interpolate(gt, size=(ph, pw), mode="nearest")
    gt_exp = gt.expand_as(pred_prob)

    focal = (
        -gt_exp * 0.5 * (1 - pred_prob) ** 2 * torch.log(pred_prob + 1e-5)
        - (1 - gt_exp) * 0.5 * pred_prob ** 2 * torch.log(1 - pred_prob + 1e-5)
    ).mean(dim=(-2, -1))

    inter = (gt_exp * pred_prob).sum(dim=(-2, -1))
    denom = gt_exp.sum(dim=(-2, -1)) + pred_prob.sum(dim=(-2, -1))
    dice = 1 - 2 * inter / (denom + 1e-5)

    seg_all = focal + dice
    seg_loss, best_idx = seg_all.min(dim=1)

    inter2 = (gt_exp * (pred_prob > 0.5).float()).sum(dim=(-2, -1))
    union2 = gt_exp.sum(dim=(-2, -1)) + (pred_prob > 0.5).float().sum(dim=(-2, -1)) - inter2
    actual_iou = inter2 / (union2 + 1e-6)
    score_loss = torch.abs(pred_scores - actual_iou.detach()).mean()

    total = seg_loss.mean() + score_loss * 0.05
    return {"seg": seg_loss.mean(), "score": score_loss, "total": total,
            "best_idx": best_idx, "actual_iou": actual_iou}
