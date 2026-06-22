"""
Evaluate DelineateAnything on AI4Boundaries test split.

DelineateAnything is a YOLO-based instance segmentation model pretrained on
FBIS-22M. Single RGB frame (August composite) input, zero-shot, no fine-tuning.

MODEL_NAME choices:
    DelineateAnything-S   ← faster (YOLO11n)
    DelineateAnything     ← more accurate (YOLO11x)
"""

# ── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_NAME     = "DelineateAnything"
DATA_ROOT      = "data/ai4boundaries/brittany"
OUT_ROOT      = "delineateanything/outputs/DelineateAnything"
SPLIT          = "test"
CONF_THRESHOLD = 0.05
IOU_THRESHOLD  = 0.3
MAX_DETECTIONS = 300
N_VISUAL       = 12


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

from shared.dataset  import AI4BoundariesSAM1Dataset
from shared.device   import best_device
from shared.metrics  import object_metrics, pixel_iou
from shared.plots     import print_stats_table, save_all_plots, SHARED_TILE_INDICES
from shared.vectorize import masks_to_gpkg
from delineateanything.model import DelineateAnythingModel

def _percentile_normalize(img: np.ndarray) -> torch.Tensor:
    """Per-band 1st–99th percentile stretch → [0, 1] float32.

    Matches the DataAnalyser normalization used by the reference DelineateAnything
    pipeline, which converts raw S2 DN to uint8 via per-band p1/p99 before feeding
    YOLO.  Without this, values sit in [0, 0.3] and YOLO treats every chip as a
    near-black image, killing all detections.
    """
    lo  = np.percentile(img, 1,  axis=(1, 2), keepdims=True)
    hi  = np.percentile(img, 99, axis=(1, 2), keepdims=True)
    out = np.clip((img - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return torch.from_numpy(out).float()


def main():
    device = best_device()
    if str(device) == "mps":
        print("MPS detected: falling back to CPU for YOLO NMS.")
        device = torch.device("cpu")

    print(f"Device: {device} | Model: {MODEL_NAME} | Data: {DATA_ROOT}/{SPLIT}")

    model = DelineateAnythingModel(
        model_name=MODEL_NAME,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        max_detections=MAX_DETECTIONS,
        device=str(device),
    )

    dataset = AI4BoundariesSAM1Dataset(root=DATA_ROOT, split=SPLIT, num_channels=3)
    print(f"Samples: {len(dataset)}")

    out_dir = ROOT / OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"results_{SPLIT}.csv"

    rows, visuals = [], []

    for idx, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        img      = sample["image"]                        # [3, H, W] float32, raw/10000
        mask     = sample["mask"]
        gt_field = (mask.numpy() > 0).astype(np.uint8)
        H, W     = img.shape[-2], img.shape[-1]

        img_norm = _percentile_normalize(img[:3].numpy())   # [3,H,W] → [0,1]

        with torch.no_grad():
            results = model(img_norm)

        pred_field  = np.zeros((H, W), dtype=np.uint8)
        pred_inst   = np.zeros((H, W), dtype=np.int32)   # per-instance label map
        inst_id     = 0
        if results and results[0].masks is not None:
            for seg in results[0].masks.data.cpu().numpy():
                seg_full = (
                    torch.nn.functional.interpolate(
                        torch.from_numpy(seg).unsqueeze(0).unsqueeze(0).float(),
                        size=(H, W), mode="nearest",
                    ).squeeze().numpy() > 0.5
                )
                pred_field = np.logical_or(pred_field, seg_full).astype(np.uint8)
                inst_id += 1
                pred_inst[seg_full & (pred_inst == 0)] = inst_id  # first-write wins

        iou = pixel_iou(
            torch.from_numpy(pred_field.astype(np.int64)),
            torch.from_numpy(gt_field.astype(np.int64)),
            num_classes=2,
        )
        # Use YOLO's own instance IDs — not cc_label on the merged mask
        obj = object_metrics(pred_inst, gt_field, use_pred_instances=True)

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

    print_stats_table(rows, f"DelineateAnything ({MODEL_NAME})", SPLIT)
    save_all_plots(rows, visuals, out_dir, f"DelineateAnything ({MODEL_NAME})",
                   n_visual=N_VISUAL, iou_key="iou_field", fixed_indices=SHARED_TILE_INDICES)
    masks_to_gpkg(dataset.samples, [v["pred"] for v in visuals],
                  out_dir / "predicted_fields.gpkg")


if __name__ == "__main__":
    main()
