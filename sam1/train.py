"""
Fine-tune SAM1 mask decoder for field boundary delineation on AI4Boundaries.

Frozen : image encoder + prompt encoder
Trained: mask decoder only (~4 M params for vit_b)

SAM1 is single-frame — sees only the August (t=5) composite (window_b).
The controlled difference vs SAM2 is that SAM2 additionally receives March
as temporal context; SAM1 cannot use it.

Note: SAM1 requires float32. MPS falls back to CPU automatically.

Edit CONFIG then run:
    python3 sam1/train.py
"""

# ── CONFIG ────────────────────────────────────────────────────────────────────

VARIANT        = "vit_b"   # vit_b | vit_l | vit_h
DATA_ROOT      = "data/ai4boundaries/france"
RESUME_DECODER = None      # e.g. "sam1/outputs/vit_b/decoder_epoch05.pt"

# ── Per-variant presets ───────────────────────────────────────────────────────
_PRESETS = {
    "vit_b": dict(
        channels=3, epochs=20, batch_size=16, lr=1e-4, weight_decay=1e-4,
        accumulation_steps=1, num_workers=4, nsel=3, max_image_size=1024, warmup_epochs=2,
        output_dir="sam1/outputs/vit_b",
    ),
    "vit_l": dict(
        channels=3, epochs=20, batch_size=4, lr=5e-5, weight_decay=1e-4,
        accumulation_steps=4, num_workers=4, nsel=3, max_image_size=1024, warmup_epochs=2,
        output_dir="sam1/outputs/vit_l",
    ),
    "vit_h": dict(
        channels=3, epochs=20, batch_size=2, lr=3e-5, weight_decay=1e-4,
        accumulation_steps=8, num_workers=4, nsel=3, max_image_size=1024, warmup_epochs=2,
        output_dir="sam1/outputs/vit_h",
    ),
}

# Scale S2 float [0-1] → SAM1's expected [0-255] range
_S2_SCALE = 3000.0

# ─────────────────────────────────────────────────────────────────────────────

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os
os.chdir(ROOT)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from shared.dataset    import AI4BoundariesSAM1Dataset
from shared.device     import amp_context, best_device, grad_scaler
from shared.losses     import sam2_loss   # same focal+dice+iou-score loss as SAM2
from shared.transforms import Sam1Transform
from sam1.model        import build_sam1


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

    # SAM1 automatic mask generator uses float64 — MPS doesn't support it
    if str(device) == "mps":
        print("MPS detected: SAM1 requires float32-only ops — falling back to CPU for training.")
        device = torch.device("cpu")

    print(f"Device: {device} | Variant: sam1_{VARIANT} | Data: {DATA_ROOT}")
    print(f"Effective batch: {cfg['batch_size']} × {accum} = {cfg['batch_size'] * accum}")

    dataset = AI4BoundariesSAM1Dataset(
        root=DATA_ROOT, split="train",
        num_channels=cfg["channels"],
        transforms=Sam1Transform(nsel=nsel, max_image_size=cfg["max_image_size"]),
    )
    loader = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], collate_fn=collate_fn, drop_last=True,
        persistent_workers=cfg["num_workers"] > 0,
        prefetch_factor=2 if cfg["num_workers"] > 0 else None,
    )
    print(f"Training samples: {len(dataset)}")

    model = build_sam1(variant=VARIANT, device=device, mode="train")

    if RESUME_DECODER and Path(RESUME_DECODER).exists():
        model.mask_decoder.load_state_dict(torch.load(RESUME_DECODER, map_location=device))
        print(f"Resumed decoder from {RESUME_DECODER}")

    trainable = list(model.mask_decoder.parameters())
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

    for epoch in range(total_e):
        pbar     = tqdm(loader, desc=f"Epoch {epoch+1}/{total_e}")
        mean_iou = 0.0
        step     = 0
        optimizer.zero_grad()

        for batch in pbar:
            if batch is None:
                continue

            img      = batch["image"].to(device)       # [B, 3, 1024, 1024]
            gt_masks = batch["gt_masks"].to(device)    # [B, nsel, H, W]
            pts      = batch["points"].to(device)      # [B, nsel, 1, 2]
            lbls     = batch["labels"].to(device)      # [B, nsel, 1]
            B, _, H, W = img.shape

            # Encode — frozen, no gradient
            with torch.no_grad():
                img_255 = torch.clamp(img * _S2_SCALE, 0, 255)
                img_pre = model.preprocess(img_255)
                image_embeddings = model.image_encoder(img_pre)   # [B, 256, 64, 64]

            with amp_context(device):
                gt_flat  = gt_masks.flatten(0, 1)   # [B*nsel, H, W]

                pts_norm = pts.float().clone()
                pts_norm[..., 0] = pts_norm[..., 0] / W * model.image_encoder.img_size
                pts_norm[..., 1] = pts_norm[..., 1] / H * model.image_encoder.img_size

                # SAM1 mask_decoder internally does repeat_interleave on image_embeddings,
                # so pass [1, 256, 64, 64] per image with [nsel, ...] prompts.
                all_masks, all_iou = [], []
                for i in range(B):
                    sparse_i, dense_i = model.prompt_encoder(
                        points=(pts_norm[i], lbls[i]), boxes=None, masks=None,
                    )
                    masks_i, iou_i = model.mask_decoder(
                        image_embeddings=image_embeddings[i:i+1],
                        image_pe=model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_i,
                        dense_prompt_embeddings=dense_i,
                        multimask_output=True,
                    )
                    all_masks.append(masks_i)
                    all_iou.append(iou_i)

                low_res_masks   = torch.cat(all_masks, dim=0)   # [B*nsel, 3, 256, 256]
                iou_predictions = torch.cat(all_iou,   dim=0)   # [B*nsel, 3]
                losses = sam2_loss(low_res_masks, iou_predictions, gt_flat.float())

            scaler.scale(losses["total"] / accum).backward()
            step += 1
            if step % accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            with torch.no_grad():
                best_iou = losses["actual_iou"].gather(1, losses["best_idx"].unsqueeze(1)).squeeze(1).mean().item()
                mean_iou = 0.99 * mean_iou + 0.01 * best_iou
            pbar.set_postfix({"loss": f"{losses['total'].item():.4f}", "mIoU": f"{mean_iou:.4f}"})

        scheduler.step()
        ckpt = out_dir / f"decoder_epoch{epoch+1:02d}.pt"
        torch.save(model.mask_decoder.state_dict(), ckpt)
        print(f"  Saved {ckpt}  |  lr={scheduler.get_last_lr()[0]:.2e}")

    final = out_dir / "decoder_final.pt"
    torch.save(model.mask_decoder.state_dict(), final)
    print(f"\nDone. Fine-tuned decoder → {final}")


if __name__ == "__main__":
    main()
