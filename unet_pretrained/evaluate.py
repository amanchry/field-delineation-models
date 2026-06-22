"""
Zero-shot evaluation of a pretrained FTW UNet on AI4Boundaries test split.

No fine-tuning — measures how well FTW pretrained weights generalise directly
to AI4Boundaries, giving a baseline against the from-scratch UNet.

Edit CONFIG then run:
    python3 unet_pretrained/evaluate.py

Available PRETRAINED_MODEL keys (see unet_pretrained/model.py):
    FTW_PRUE_EFNET_B5      ← recommended
    FTW_PRUE_EFNET_B5_CCBY
    FTW_PRUE_EFNET_B3      FTW_PRUE_EFNET_B3_CCBY
    FTW_PRUE_EFNET_B7      FTW_PRUE_EFNET_B7_CCBY
    FTW_v2_3_Class_FULL_multiWindow
    FTW_v1_3_Class_FULL    FTW_v1_3_Class_CCBY
"""

# ── CONFIG ────────────────────────────────────────────────────────────────────

PRETRAINED_MODEL = "FTW_PRUE_EFNET_B5"
WEIGHTS_DIR      = "unet_pretrained/weights"
DATA_ROOT        = "data/ai4boundaries/brittany"
SPLIT            = "test"
N_VISUAL         = 12

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

from shared.dataset        import AI4BoundariesDataset
from shared.device         import best_device
from shared.metrics        import object_metrics, pixel_iou
from shared.plots          import print_stats_table, save_all_plots, SHARED_TILE_INDICES
from shared.vectorize      import masks_to_gpkg
from unet_pretrained.model import load_pretrained_model


def main():
    device = best_device()
    print(f"Device: {device} | Model: {PRETRAINED_MODEL} | Data: {DATA_ROOT}")

    model, num_classes = load_pretrained_model(PRETRAINED_MODEL, WEIGHTS_DIR, str(device))

    dataset = AI4BoundariesDataset(DATA_ROOT, split=SPLIT)
    loader  = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
    print(f"Samples: {len(dataset)}")

    out_dir = ROOT / "unet_pretrained" / "outputs" / PRETRAINED_MODEL
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"results_{SPLIT}.csv"

    rows, visuals = [], []
    all_iou = {c: [] for c in range(num_classes)}

    class_names = {0: "background", 1: "field", 2: "boundary"}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            imgs  = batch["image"].to(device)
            masks = batch["mask"]
            preds = model(imgs).argmax(dim=1).cpu()

            for i in range(imgs.shape[0]):
                iou = pixel_iou(preds[i], masks[i], num_classes)
                obj = object_metrics(preds[i].numpy(), masks[i].numpy())

                for c in range(num_classes):
                    all_iou[c].append(iou[c])

                row = {
                    "mean_iou":      iou["mean"],
                    "obj_precision": obj["precision"],
                    "obj_recall":    obj["recall"],
                    "obj_f1":        obj["f1"],
                }
                for c in range(num_classes):
                    row[f"iou_class{c}"] = iou[c]
                rows.append(row)

                # August composite: channels 4=B4(R), 5=B3(G), 6=B2(B)
                rgb = imgs[i, 4:7].cpu().numpy().transpose(1, 2, 0)
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

    # Per-class IoU summary
    print(f"\n=== UNet Pretrained ({PRETRAINED_MODEL}) — {SPLIT} split  ({len(rows)} samples) ===")
    for c in range(num_classes):
        print(f"  IoU {class_names.get(c, f'class{c}'):<12}: {np.mean(all_iou[c]):.4f}")
    print(f"  Mean IoU       : {np.mean([r['mean_iou'] for r in rows]):.4f}")

    print_stats_table(rows, f"UNet Pretrained ({PRETRAINED_MODEL})", SPLIT, iou_key="iou_class1")
    save_all_plots(rows, visuals, out_dir, f"UNet Pretrained ({PRETRAINED_MODEL})",
                   n_visual=N_VISUAL, iou_key="iou_class1", fixed_indices=SHARED_TILE_INDICES)
    masks_to_gpkg(dataset.samples, [v["pred"] for v in visuals],
                  out_dir / "predicted_fields.gpkg")


if __name__ == "__main__":
    main()
