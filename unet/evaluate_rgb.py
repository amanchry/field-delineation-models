"""
Evaluate a trained RGB-only UNet checkpoint on the AI4Boundaries test split.

This script pairs with train_rgb.py — uses 6-band input (B4/B3/B2 × March+August),
no NIR channel.

Edit CONFIG then run:
    python3 unet/evaluate_rgb.py
"""

# ── CONFIG ────────────────────────────────────────────────────────────────────

DATA_ROOT   = "data/ai4boundaries/brittany"
SPLIT       = "test"
BACKBONE    = "efficientnet-b5"
IN_CHANNELS = 6        # 3 RGB bands × 2 time windows
NUM_CLASSES = 3
CHECKPOINT  = "unet/outputs/efficientnet-b5_rgb/best.pt"
OUT_DIR     = "unet/outputs/efficientnet-b5_rgb"
N_VISUAL    = 4

# ─────────────────────────────────────────────────────────────────────────────

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os
os.chdir(ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from shared.dataset import AI4BoundariesDataset
from shared.device  import best_device
from shared.metrics import object_metrics, pixel_iou
from shared.plots   import print_stats_table, save_all_plots, SHARED_TILE_INDICES
from unet.model     import build_unet


def main():
    ckpt_path = ROOT / CHECKPOINT
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        print("  Train first:  python3 unet/train_rgb.py")
        return

    device = best_device()
    print(f"Device: {device} | Checkpoint: {ckpt_path} | Split: {SPLIT}")
    print(f"Input channels: {IN_CHANNELS} (RGB × 2 windows, no NIR)")

    model = build_unet(
        backbone=BACKBONE, in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES, pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
    model.eval()

    # num_channels=3 → dataset returns [6, 256, 256] (RGB × March + August)
    dataset = AI4BoundariesDataset(DATA_ROOT, split=SPLIT, num_channels=3)
    loader  = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
    print(f"Samples: {len(dataset)}")

    out_dir = ROOT / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"results_{SPLIT}.csv"

    rows, visuals = [], []
    all_iou = {c: [] for c in range(NUM_CLASSES)}
    class_names = {0: "background", 1: "field", 2: "boundary"}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            imgs  = batch["image"].to(device)
            masks = batch["mask"]
            preds = model(imgs).argmax(dim=1).cpu()

            for i in range(imgs.shape[0]):
                iou = pixel_iou(preds[i], masks[i], NUM_CLASSES)
                obj = object_metrics(preds[i].numpy(), masks[i].numpy())

                for c in range(NUM_CLASSES):
                    all_iou[c].append(iou[c])

                row = {
                    "mean_iou":      iou["mean"],
                    "obj_precision": obj["precision"],
                    "obj_recall":    obj["recall"],
                    "obj_f1":        obj["f1"],
                }
                for c in range(NUM_CLASSES):
                    row[f"iou_class{c}"] = iou[c]
                rows.append(row)

                # August RGB: channels 3=B4(R), 4=B3(G), 5=B2(B)
                rgb = imgs[i, 3:6].cpu().numpy().transpose(1, 2, 0)
                visuals.append({
                    "rgb":  rgb,
                    "gt":   (masks[i].numpy() == 1).astype(np.uint8),
                    "pred": (preds[i].numpy()  == 1).astype(np.uint8),
                    "iou":  iou[1],
                    "f1":   obj["f1"],
                })

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results CSV → {out_csv}")

    print(f"\n=== UNet-RGB ({BACKBONE}) — {SPLIT} split  ({len(rows)} samples) ===")
    for c in range(NUM_CLASSES):
        print(f"  IoU {class_names.get(c, f'class{c}'):<12}: {np.mean(all_iou[c]):.4f}")
    print(f"  Mean IoU       : {np.mean([r['mean_iou'] for r in rows]):.4f}")

    print_stats_table(rows, f"UNet-RGB ({BACKBONE})", SPLIT, iou_key="iou_class1")
    save_all_plots(rows, visuals, out_dir, f"UNet-RGB ({BACKBONE})",
                   n_visual=N_VISUAL, iou_key="iou_class1", fixed_indices=SHARED_TILE_INDICES)


if __name__ == "__main__":
    main()
