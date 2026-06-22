"""
Evaluate a fine-tuned SAM2 decoder on the AI4Boundaries test split.

For each sample:
  1. Encode window_a (March) as temporal context (no grad).
  2. Auto-generate masks on window_b (August) with the fine-tuned decoder.
  3. Merge masks into a binary field prediction.
  4. Compute pixel IoU and object Precision / Recall / F1.
"""

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
import torch.nn.functional as F
from tqdm import tqdm

from shared.dataset import AI4BoundariesSAM2Dataset
from shared.device  import best_device
from shared.metrics import object_metrics, pixel_iou
from shared.plots     import print_stats_table, save_all_plots, SHARED_TILE_INDICES
from shared.vectorize import masks_to_gpkg
from sam2.model     import build_sam2, load_sam2_mask_generator


# ── CONFIG ────────────────────────────────────────────────────────────────────

VARIANT      = "base_plus"
DECODER_PATH = "sam2/outputs/base_plus_v2/decoder_final.pt"   # None = base SAM2
DATA_ROOT    = "data/ai4boundaries/brittany"
OUT_ROOT    = "sam2/outputs/base_plus_v2"
SPLIT        = "test"
CONF_THRESH  = 0.5
N_VISUAL     = 4



def main():
    device = best_device()
    print(f"Device: {device} | Variant: sam2_{VARIANT} | Data: {DATA_ROOT}")

    model = build_sam2(variant=VARIANT, device=device, mode="eval")

    if DECODER_PATH and Path(DECODER_PATH).exists():
        model.sam_mask_decoder.load_state_dict(torch.load(DECODER_PATH, map_location=device))
        print(f"Loaded fine-tuned decoder from {DECODER_PATH}")
    else:
        print("No fine-tuned decoder — evaluating base SAM2.")
    model.eval()

    generator = load_sam2_mask_generator(model)

    dataset = AI4BoundariesSAM2Dataset(root=DATA_ROOT, split=SPLIT, num_channels=3)
    print(f"Samples: {len(dataset)}")

    out_dir = ROOT / OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"results_{SPLIT}.csv"

    rows, visuals = [], []

    for idx, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        img_a     = sample["window_a"]
        img_b     = sample["window_b"]
        inst_mask = sample["mask"]
        gt_field  = (inst_mask.numpy() > 0).astype(np.uint8)
        H, W      = img_a.shape[-2], img_a.shape[-1]

        def to_rgb_uint8(t):
            return np.clip(t[:3].numpy() * 3000, 0, 255).astype(np.uint8).transpose(1, 2, 0)

        img_b_uint8 = to_rgb_uint8(img_b)

        with torch.no_grad():
            img_a_sq = F.interpolate(img_a.unsqueeze(0), size=(1024, 1024),
                                     mode="bilinear", align_corners=False)
            model.forward_image(img_a_sq.to(device))
            masks = generator.generate(img_b_uint8)

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
            "rgb":  img_b[:3].numpy().transpose(1, 2, 0),  # raw [0,~0.3], stretched in plot
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

    print_stats_table(rows, f"SAM2 Fine-tuned ({VARIANT})", SPLIT)
    save_all_plots(rows, visuals, out_dir, f"SAM2 Fine-tuned ({VARIANT})",
                   n_visual=N_VISUAL, iou_key="iou_field", fixed_indices=SHARED_TILE_INDICES)
    masks_to_gpkg(dataset.samples, [v["pred"] for v in visuals],
                  out_dir / "predicted_fields.gpkg")


if __name__ == "__main__":
    main()
