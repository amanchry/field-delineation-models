"""
Train UNet on AI4Boundaries data — RGB-only variant.

Architecture : UNet + EfficientNet backbone
Input        : 6-band Sentinel-2 (March + August, RGB only: B4/B3/B2 each)
               NIR (B8) is excluded so the model is comparable to single-modality
               RGB approaches (SAM1, SAM2, DelineateAnything).
Output       : 3-class segmentation  (0=background  1=field  2=boundary)

Checkpoints saved to: unet/outputs/<BACKBONE>_rgb/
"""

# ── CONFIG ────────────────────────────────────────────────────────────────────

DATA_ROOT = "data/ai4boundaries/brittany"

# Backbone: "efficientnet-b3" (fast, ~12M params) | "efficientnet-b5" (default, ~30M params)
BACKBONE = "efficientnet-b5"

# Resume from an existing checkpoint, or None to start fresh
RESUME_FROM = None    # e.g. "unet/outputs/efficientnet-b5_rgb/last.pt"

OUT_ROOT=f"unet/outputs/{BACKBONE}_rgb"


# ── Per-backbone presets ──────────────────────────────────────────────────────
_PRESETS = {
    "efficientnet-b3": dict(
        in_channels  = 6,     # 3 RGB bands × 2 time windows
        num_classes  = 3,
        pretrained   = True,
        epochs       = 100,
        batch_size   = 16,
        lr           = 1e-4,
        weight_decay = 1e-4,
        num_workers  = 4,
    ),
    "efficientnet-b5": dict(
        in_channels  = 6,     # 3 RGB bands × 2 time windows
        num_classes  = 3,
        pretrained   = True,
        epochs       = 100,
        batch_size   = 8,
        lr           = 1e-4,
        weight_decay = 1e-4,
        num_workers  = 4,
    ),
}

# ─────────────────────────────────────────────────────────────────────────────

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os
os.chdir(ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from shared.dataset    import AI4BoundariesDataset
from shared.device     import amp_context, best_device, grad_scaler
from shared.losses     import dice_loss
from shared.metrics    import pixel_iou
from shared.transforms import UNetTransform
from unet.model        import build_unet


def main():
    cfg     = _PRESETS[BACKBONE]
    out_dir = ROOT / OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    device = best_device()
    print(f"Device: {device} | Backbone: {BACKBONE}_rgb | Data: {DATA_ROOT}")
    print(f"Input channels: {cfg['in_channels']} (RGB × 2 windows, no NIR)")

    train_ds = AI4BoundariesDataset(DATA_ROOT, split="train",
                                    transforms=UNetTransform(), num_channels=3)
    val_ds   = AI4BoundariesDataset(DATA_ROOT, split="val", num_channels=3)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"],
    )
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    model = build_unet(
        backbone=BACKBONE,
        in_channels=cfg["in_channels"],
        num_classes=cfg["num_classes"],
        pretrained=cfg["pretrained"],
    ).to(device)

    if RESUME_FROM and Path(RESUME_FROM).exists():
        model.load_state_dict(torch.load(RESUME_FROM, map_location=device))
        print(f"Resumed from {RESUME_FROM}")

    ce_weight = torch.tensor([0.3, 1.0, 5.0]).to(device)
    ce_loss   = nn.CrossEntropyLoss(weight=ce_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    scaler    = grad_scaler(device)
    best_miou = 0.0
    log_path  = out_dir / "train_log.csv"
    log_file  = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=["epoch", "train_loss", "train_miou", "val_loss", "val_miou"])
    log_writer.writeheader()

    for epoch in range(cfg["epochs"]):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = train_iou = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']} [train]", leave=False):
            img  = batch["image"].to(device)
            mask = batch["mask"].to(device)
            optimizer.zero_grad()
            with amp_context(device):
                logits = model(img)
                loss   = ce_loss(logits, mask)
                soft   = torch.softmax(logits, dim=1)
                dice_weights = [0.2, 0.4, 1.4]
                for c, dw in enumerate(dice_weights):
                    loss = loss + dw * dice_loss(soft[:, c], (mask == c).float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                for i in range(img.shape[0]):
                    train_iou += pixel_iou(pred[i], mask[i], cfg["num_classes"])["mean"]
        scheduler.step()
        train_loss /= len(train_loader)
        train_iou  /= len(train_ds)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = val_iou = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']} [val]  ", leave=False):
                img  = batch["image"].to(device)
                mask = batch["mask"].to(device)
                with amp_context(device):
                    logits = model(img)
                    loss   = ce_loss(logits, mask)
                val_loss += loss.item()
                pred = logits.argmax(dim=1)
                for i in range(img.shape[0]):
                    val_iou += pixel_iou(pred[i], mask[i], cfg["num_classes"])["mean"]
        val_loss /= len(val_loader)
        val_iou  /= len(val_ds)

        print(
            f"Epoch {epoch+1:03d} | "
            f"train_loss={train_loss:.4f}  train_mIoU={train_iou:.4f} | "
            f"val_loss={val_loss:.4f}  val_mIoU={val_iou:.4f}"
        )

        log_writer.writerow({"epoch": epoch + 1, "train_loss": round(train_loss, 6),
                              "train_miou": round(train_iou, 6), "val_loss": round(val_loss, 6),
                              "val_miou": round(val_iou, 6)})
        log_file.flush()

        torch.save(model.state_dict(), out_dir / "last.pt")
        if val_iou > best_miou:
            best_miou = val_iou
            torch.save(model.state_dict(), out_dir / "best.pt")
            print(f"  New best saved  (val_mIoU={best_miou:.4f})")

    log_file.close()
    print(f"\nDone. Best checkpoint → {out_dir / 'best.pt'}")
    print(f"Training log        → {log_path}")


if __name__ == "__main__":
    main()
