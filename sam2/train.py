"""
Fine-tune SAM2 mask decoder for field boundary delineation on AI4Boundaries.

Frozen : image encoder + prompt encoder
Trained: mask decoder only (~4 M params)

Two Sentinel-2 time windows treated as a 2-frame video:
  window_a (March,  t=0) → temporal context (no gradient)
  window_b (August, t=5) → prediction frame (gradients flow through decoder)

Edit CONFIG then run:
    python3 sam2/train.py
"""

# ── CONFIG ────────────────────────────────────────────────────────────────────

VARIANT    = "small"   # tiny | small | base_plus | large
DATA_ROOT  = "data/ai4boundaries/france"
RESUME_DECODER = None      # e.g. "sam2/outputs/base_plus/decoder_epoch05.pt"

# ── Per-variant presets ───────────────────────────────────────────────────────
_PRESETS = {
    "tiny": dict(
        channels=3, epochs=50, batch_size=16, lr=1e-4, weight_decay=1e-4,
        accumulation_steps=1, num_workers=4, nsel=3, max_image_size=1024, warmup_epochs=2,
        output_dir="sam2/outputs/tiny",
    ),
    "small": dict(
        channels=3, epochs=50, batch_size=8, lr=1e-4, weight_decay=1e-4,
        accumulation_steps=2, num_workers=4, nsel=3, max_image_size=1024, warmup_epochs=2,
        output_dir="sam2/outputs/small",
    ),
    "base_plus": dict(
        channels=3, epochs=50, batch_size=4, lr=5e-5, weight_decay=1e-4,
        accumulation_steps=4, num_workers=4, nsel=3, max_image_size=1024, warmup_epochs=2,
        output_dir="sam2/outputs/base_plus",
    ),
    "large": dict(
        channels=3, epochs=50, batch_size=2, lr=3e-5, weight_decay=1e-4,
        accumulation_steps=8, num_workers=4, nsel=3, max_image_size=1024, warmup_epochs=2,
        output_dir="sam2/outputs/large",
    ),
}

# ─────────────────────────────────────────────────────────────────────────────

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os
os.chdir(ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from shared.dataset    import AI4BoundariesSAM2Dataset
from shared.device     import amp_context, best_device, grad_scaler
from shared.losses     import sam2_loss
from shared.transforms import Sam2Transform
from sam2.model        import build_sam2


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return torch.utils.data.dataloader.default_collate(batch) if batch else None


def main():
    cfg     = _PRESETS[VARIANT]
    nsel    = cfg["nsel"]
    accum   = cfg["accumulation_steps"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = best_device()

    print(f"Device: {device} | Variant: sam2_{VARIANT} | Data: {DATA_ROOT}")
    print(f"Effective batch: {cfg['batch_size']} × {accum} = {cfg['batch_size'] * accum}")

    dataset = AI4BoundariesSAM2Dataset(
        root=DATA_ROOT, split="train",
        num_channels=cfg["channels"],
        transforms=Sam2Transform(nsel=nsel, max_image_size=cfg["max_image_size"]),
    )
    loader = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], collate_fn=collate_fn, drop_last=True,
        persistent_workers=cfg["num_workers"] > 0,
        prefetch_factor=2 if cfg["num_workers"] > 0 else None,
    )
    print(f"Training samples: {len(dataset)}")

    model = build_sam2(variant=VARIANT, device=device, mode="train")

    if RESUME_DECODER and Path(RESUME_DECODER).exists():
        model.sam_mask_decoder.load_state_dict(torch.load(RESUME_DECODER, map_location=device))
        print(f"Resumed decoder from {RESUME_DECODER}")

    trainable = list(model.sam_mask_decoder.parameters())
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler    = grad_scaler(device)
    warmup    = cfg.get("warmup_epochs", 2)
    total_e   = cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_e - warmup, eta_min=cfg["lr"] * 0.01),
        ],
        milestones=[warmup],
    )

    best_loss  = float("inf")
    best_epoch = 0

    for epoch in range(total_e):
        pbar           = tqdm(loader, desc=f"Epoch {epoch+1}/{total_e}")
        mean_iou       = 0.0
        epoch_loss_sum = 0.0
        step           = 0
        optimizer.zero_grad()

        for batch in pbar:
            if batch is None:
                continue

            img_a    = batch["img_a"].to(device)
            img_b    = batch["img_b"].to(device)
            gt_masks = batch["gt_masks"].to(device)
            pts      = batch["points"].to(device)
            lbls     = batch["labels"].to(device)
            B, _, H, W = img_b.shape

            with torch.no_grad():
                model.forward_image(img_a)

            with amp_context(device):
                feats = model.forward_image(img_b)
                vis   = feats["vision_features"]
                vis   = vis.unsqueeze(1).expand(-1, nsel, -1, -1, -1).flatten(0, 1)

                high_res = None
                if getattr(model, "use_high_res_features_in_sam", False):
                    high_res = [
                        f.unsqueeze(1).expand(-1, nsel, -1, -1, -1).flatten(0, 1)
                        for f in feats["backbone_fpn"][:2]
                    ]

                pts_flat  = pts.flatten(0, 1).float()
                lbls_flat = lbls.flatten(0, 1)
                gt_flat   = gt_masks.flatten(0, 1)

                pts_norm = pts_flat.clone()
                pts_norm[..., 0] = pts_norm[..., 0] / W * model.image_size
                pts_norm[..., 1] = pts_norm[..., 1] / H * model.image_size

                # Points-only prompts — matches SamAutomaticMaskGenerator at eval time.
                # GT mask prompts were removed: they caused the decoder to learn to
                # "refine an answer" rather than "segment from scratch", which collapsed
                # performance under automatic (no-prompt) evaluation.
                sparse, dense = model.sam_prompt_encoder(
                    points=(pts_norm, lbls_flat), boxes=None, masks=None
                )

                low_res_masks, pred_scores, _, _ = model.sam_mask_decoder(
                    image_embeddings=vis,
                    image_pe=model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=True,
                    repeat_image=False,
                    high_res_features=high_res,
                )
                pred   = F.interpolate(low_res_masks, size=(H, W), mode="bilinear", align_corners=False)
                losses = sam2_loss(pred, pred_scores, gt_flat.float())

            scaler.scale(losses["total"] / accum).backward()
            step += 1
            if step % accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            with torch.no_grad():
                best_iou = losses["actual_iou"].gather(1, losses["best_idx"].unsqueeze(1)).squeeze(1).mean().item()
                mean_iou = 0.99 * mean_iou + 0.01 * best_iou
                epoch_loss_sum += losses["total"].item()
            pbar.set_postfix({"loss": f"{losses['total'].item():.4f}", "mIoU": f"{mean_iou:.4f}"})

        scheduler.step()
        epoch_loss_avg = epoch_loss_sum / max(step, 1)
        ckpt = out_dir / f"decoder_epoch{epoch+1:02d}.pt"
        torch.save(model.sam_mask_decoder.state_dict(), ckpt)

        # Save best checkpoint separately
        if epoch == 0 or epoch_loss_avg < best_loss:
            best_loss  = epoch_loss_avg
            best_epoch = epoch + 1
            torch.save(model.sam_mask_decoder.state_dict(), out_dir / "decoder_best.pt")
            best_tag = "  ← best"
        else:
            best_tag = ""
        print(f"  Epoch {epoch+1:02d}  loss={epoch_loss_avg:.4f}  mIoU={mean_iou:.4f}"
              f"  lr={scheduler.get_last_lr()[0]:.2e}{best_tag}")

    final = out_dir / "decoder_final.pt"
    torch.save(model.sam_mask_decoder.state_dict(), final)
    print(f"\nBest checkpoint: epoch {best_epoch}  → {out_dir}/decoder_best.pt")
    print(f"\nDone. Fine-tuned decoder → {final}")


if __name__ == "__main__":
    main()
