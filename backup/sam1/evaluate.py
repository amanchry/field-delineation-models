"""
Evaluate a fine-tuned SAM1 decoder on the AI4Boundaries test split.

Loads the decoder checkpoint produced by train.py, generates masks with
SamAutomaticMaskGenerator, and reports pixel IoU + object P/R/F1.

Edit CONFIG then run:
    python3 sam1/evaluate.py
"""

# ── CONFIG ────────────────────────────────────────────────────────────────────

VARIANT      = "vit_b"
DECODER_PATH = "sam1/outputs/vit_b/decoder_final.pt"
DATA_ROOT    = "data/ai4boundaries/brittany"
SPLIT        = "test"
CONF_THRESH  = 0.5
N_VISUAL     = 12

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
from tqdm import tqdm

from shared.dataset import AI4BoundariesSAM1Dataset
from shared.device  import best_device
from shared.metrics import object_metrics, pixel_iou
from shared.plots     import print_stats_table, save_all_plots, SHARED_TILE_INDICES
from shared.vectorize import masks_to_gpkg
from sam1.model     import build_sam1

_S2_SCALE = 3000.0


def main():
    device = best_device()
    if str(device) == "mps":
        print("MPS detected: SAM1 automatic mask generator requires CPU — falling back.")
        device = torch.device("cpu")

    print(f"Device: {device} | Variant: sam1_{VARIANT} | Data: {DATA_ROOT}")

    if not Path(DECODER_PATH).exists():
        print(f"ERROR: decoder checkpoint not found: {DECODER_PATH}")
        print("  Train first:  python3 sam1/train.py")
        return

    model = build_sam1(variant=VARIANT, device=device, mode="eval")
    model.mask_decoder.load_state_dict(torch.load(DECODER_PATH, map_location=device))
    model.eval()
    print(f"Loaded fine-tuned decoder from {DECODER_PATH}")

    try:
        from segment_anything import SamAutomaticMaskGenerator
        generator = SamAutomaticMaskGenerator(model, pred_iou_thresh=CONF_THRESH)
    except ImportError:
        raise ImportError(
            "segment-anything package required.\n"
            "Install: pip install git+https://github.com/facebookresearch/segment-anything.git"
        )

    dataset = AI4BoundariesSAM1Dataset(root=DATA_ROOT, split=SPLIT, num_channels=3)
    print(f"Samples: {len(dataset)}")

    out_dir = ROOT / "sam1" / "outputs" / VARIANT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"results_{SPLIT}.csv"

    rows, visuals = [], []

    for idx, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        img      = sample["image"]                              # [3, H, W] float32
        mask     = sample["mask"]
        gt_field = (mask.numpy() > 0).astype(np.uint8)
        H, W     = img.shape[-2], img.shape[-1]

        img_uint8 = np.clip(img.numpy() * _S2_SCALE, 0, 255).astype(np.uint8).transpose(1, 2, 0)

        with torch.no_grad():
            masks = generator.generate(img_uint8)

        pred_field = np.zeros((H, W), dtype=np.uint8)
        for m in masks:
            if m.get("predicted_iou", 1.0) >= CONF_THRESH:
                pred_field = np.logical_or(pred_field, m["segmentation"]).astype(np.uint8)

        iou = pixel_iou(
            torch.from_numpy(pred_field.astype(np.int64)),
            torch.from_numpy(gt_field.astype(np.int64)),
            num_classes=2,
        )
        obj = object_metrics(pred_field, gt_field)

        rows.append({
            "sample_idx":    idx,
            "iou_field":     iou.get(1, 0.0),
            "obj_precision": obj["precision"],
            "obj_recall":    obj["recall"],
            "obj_f1":        obj["f1"],
        })
        visuals.append({
            "rgb":  img[:3].numpy().transpose(1, 2, 0),   # raw [0,~0.3], stretched in plot
            "gt":   gt_field,
            "pred": pred_field,
            "iou":  iou.get(1, 0.0),
            "f1":   obj["f1"],
        })

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results CSV → {out_csv}")

    print_stats_table(rows, f"SAM1 Fine-tuned ({VARIANT})", SPLIT)
    save_all_plots(rows, visuals, out_dir, f"SAM1 Fine-tuned ({VARIANT})",
                   n_visual=N_VISUAL, iou_key="iou_field", fixed_indices=SHARED_TILE_INDICES)
    masks_to_gpkg(dataset.samples, [v["pred"] for v in visuals],
                  out_dir / "predicted_fields.gpkg")


if __name__ == "__main__":
    main()
