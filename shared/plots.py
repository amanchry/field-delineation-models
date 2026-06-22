"""
Shared evaluation plots for all model evaluate scripts.

Usage in any evaluate.py:

    from shared.plots import print_stats_table, save_all_plots

    # rows  — list of dicts with at least iou_key, obj_precision, obj_recall, obj_f1
    # visuals — list of dicts: {rgb, gt, pred, iou, f1}
    #   rgb : np.ndarray [H, W, 3] float32  (raw values; percentile-stretched internally)
    #   gt  : np.ndarray [H, W]   uint8     (binary field ground truth)
    #   pred: np.ndarray [H, W]   uint8     (binary field prediction)

    print_stats_table(rows, model_label="MyModel", split="test", iou_key="iou_field")
    save_all_plots(rows, visuals, out_dir, model_label="MyModel", n_visual=12, iou_key="iou_field")

Outputs written to out_dir/:
    metrics_summary.png    bar chart mean ± std
    iou_histogram.png      per-sample IoU distribution
    metrics_boxplot.png    box plots for all four metrics
    pr_scatter.png         Precision vs Recall coloured by IoU
    confusion_matrix.png   pixel-level TP / FP / FN / TN
    visual_grid.png        N × 4 grid: RGB | GT | Prediction | Error map
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path

# Dataset-order indices used in visual_grid.png for ALL models so tiles are comparable.
# Change these to indices that are interesting in your test split.
SHARED_TILE_INDICES = [0, 2,4, 10]


# ── Internal helper ───────────────────────────────────────────────────────────

def _savefig(fig, path: Path, dpi: int = 150):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Stats table ───────────────────────────────────────────────────────────────

def print_stats_table(rows, model_label: str, split: str, iou_key: str = "iou_field"):
    """Print mean / std / median / P25 / P75 for the four core metrics."""
    ious  = [float(r[iou_key])         for r in rows]
    precs = [float(r["obj_precision"]) for r in rows]
    recs  = [float(r["obj_recall"])    for r in rows]
    f1s   = [float(r["obj_f1"])        for r in rows]

    print(f"\n{'='*62}")
    print(f"  {model_label}  —  {len(rows)} {split} samples")
    print(f"{'='*62}")
    print(f"  {'Metric':<20} {'Mean':>7}  {'Std':>7}  {'Median':>7}  {'P25':>7}  {'P75':>7}")
    print(f"  {'-'*60}")
    for name, vals in [("Field IoU", ious), ("Precision", precs),
                       ("Recall",    recs),  ("F1",        f1s)]:
        a = np.array(vals)
        print(f"  {name:<20} {a.mean():>7.4f}  {a.std():>7.4f}  "
              f"{np.median(a):>7.4f}  {np.percentile(a, 25):>7.4f}  {np.percentile(a, 75):>7.4f}")
    print(f"{'='*62}\n")


# ── Individual plot functions ─────────────────────────────────────────────────

def plot_summary_bar(rows, path: Path, model_label: str, iou_key: str = "iou_field"):
    metrics = {
        "Field IoU": np.mean([float(r[iou_key])         for r in rows]),
        "Precision": np.mean([float(r["obj_precision"])  for r in rows]),
        "Recall":    np.mean([float(r["obj_recall"])     for r in rows]),
        "F1":        np.mean([float(r["obj_f1"])         for r in rows]),
    }
    stds = {
        "Field IoU": np.std([float(r[iou_key])         for r in rows]),
        "Precision": np.std([float(r["obj_precision"])  for r in rows]),
        "Recall":    np.std([float(r["obj_recall"])     for r in rows]),
        "F1":        np.std([float(r["obj_f1"])         for r in rows]),
    }
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(metrics.keys(), metrics.values(), yerr=stds.values(),
                  color=colors, capsize=5, alpha=0.85, edgecolor="white")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"{model_label}  —  {len(rows)} test samples")
    for bar, v in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    _savefig(fig, path)


def plot_iou_histogram(rows, path: Path, iou_key: str = "iou_field"):
    ious = [float(r[iou_key]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ious, bins=25, color="#2196F3", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(ious), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {np.mean(ious):.3f}")
    ax.axvline(np.median(ious), color="orange", linestyle="--", linewidth=1.5,
               label=f"Median = {np.median(ious):.3f}")
    ax.set_xlabel("Field IoU")
    ax.set_ylabel("Count")
    ax.set_title("Per-sample Field IoU distribution")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    _savefig(fig, path)


def plot_metrics_boxplot(rows, path: Path, iou_key: str = "iou_field"):
    data = {
        "Field IoU": [float(r[iou_key])         for r in rows],
        "Precision": [float(r["obj_precision"])  for r in rows],
        "Recall":    [float(r["obj_recall"])     for r in rows],
        "F1":        [float(r["obj_f1"])         for r in rows],
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot(data.values(), tick_labels=data.keys(), patch_artist=True,
                    medianprops={"color": "white", "linewidth": 2})
    for patch, color in zip(bp["boxes"], ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Metric distributions across test set")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    _savefig(fig, path)


def plot_pr_scatter(rows, path: Path, iou_key: str = "iou_field"):
    prec = np.array([float(r["obj_precision"]) for r in rows])
    rec  = np.array([float(r["obj_recall"])    for r in rows])
    iou  = np.array([float(r[iou_key])         for r in rows])
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(rec, prec, c=iou, cmap="RdYlGn", vmin=0, vmax=1,
                    alpha=0.6, s=20, edgecolors="none")
    fig.colorbar(sc, ax=ax, label="Field IoU")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Precision vs Recall (coloured by IoU)")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    _savefig(fig, path)


def plot_confusion_matrix(visuals, path: Path):
    """Pixel-level binary confusion matrix accumulated over all samples."""
    tp = fp = fn = tn = 0
    for v in visuals:
        p, g = v["pred"].astype(bool), v["gt"].astype(bool)
        tp += int(( p &  g).sum())
        fp += int(( p & ~g).sum())
        fn += int((~p &  g).sum())
        tn += int((~p & ~g).sum())
    total = tp + fp + fn + tn
    cm = np.array([[tn, fp], [fn, tp]], dtype=np.float64) / total * 100
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
    fig.colorbar(im, ax=ax, label="% of total pixels")
    labels = ["Background", "Field"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground Truth")
    ax.set_title("Pixel-level confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:.1f}%", ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if cm[i, j] > 50 else "black")
    _savefig(fig, path)


def _get_boundary(mask: np.ndarray) -> np.ndarray:
    """Return a boolean mask of the 1-pixel inner boundary of a binary mask."""
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(mask, iterations=1)
    return mask.astype(bool) & ~eroded


def plot_visual_grid(visuals, n: int, path: Path, fixed_indices=None):
    """N samples × 4 columns: RGB | Ground Truth | Prediction | Error map.

    fixed_indices: list of dataset-order indices to show (same across all models).
                   If None, falls back to IoU-sorted spread.
    """
    if fixed_indices is not None:
        picks = [visuals[i] for i in fixed_indices if i < len(visuals)]
    else:
        sorted_v = sorted(visuals, key=lambda x: x["iou"])
        step  = max(1, len(sorted_v) // n)
        picks = [sorted_v[i * step] for i in range(min(n, len(sorted_v)))]

    COL_TITLES   = ["Sentinel-2 RGB", "Ground Truth", "Prediction", "Error Map"]
    LEGEND_ITEMS = [
        ("#4CAF50", "TP — correct field"),
        ("#F44336", "FP — false alarm"),
        ("#2196F3", "FN — missed field"),
    ]

    ncols, nrows = 4, len(picks)
    cell_px = 2.6
    fig = plt.figure(figsize=(ncols * cell_px, nrows * cell_px + 0.5))
    gs  = gridspec.GridSpec(
        nrows, ncols, figure=fig,
        hspace=0.03, wspace=0.03,
        top=0.93, bottom=0.055, left=0.01, right=0.99,
    )

    # Pre-create all axes so indexing is clean
    axes = [[fig.add_subplot(gs[r, c]) for c in range(ncols)] for r in range(nrows)]

    # Column titles on the top row only
    for col, title in enumerate(COL_TITLES):
        axes[0][col].set_title(title, fontsize=12, fontweight="bold", pad=3)

    for row, v in enumerate(picks):
        rgb_raw = v["rgb"]
        p2, p98 = np.percentile(rgb_raw, 2), np.percentile(rgb_raw, 98)
        rgb  = np.clip((rgb_raw - p2) / (p98 - p2 + 1e-6), 0, 1)
        gt   = v["gt"]
        pred = v["pred"]

        # GT — bright cyan fill + white boundary outline
        gt_vis = rgb.copy()
        gt_vis[gt.astype(bool)] = (
            gt_vis[gt.astype(bool)] * 0.25 + np.array([0.0, 0.95, 0.95]) * 0.75
        )
        gt_boundary = _get_boundary(gt)
        gt_vis[gt_boundary] = [1.0, 1.0, 1.0]   # white outline

        # Prediction — bright magenta fill + yellow boundary outline
        pred_vis = rgb.copy()
        pred_vis[pred.astype(bool)] = (
            pred_vis[pred.astype(bool)] * 0.25 + np.array([1.0, 0.08, 0.58]) * 0.75
        )
        pred_boundary = _get_boundary(pred)
        pred_vis[pred_boundary] = [1.0, 0.95, 0.0]  # yellow outline
        err = np.zeros((*gt.shape, 3), dtype=np.float32)
        err[ gt.astype(bool) &  pred.astype(bool)] = [0.30, 0.69, 0.31]
        err[~gt.astype(bool) &  pred.astype(bool)] = [0.96, 0.26, 0.21]
        err[ gt.astype(bool) & ~pred.astype(bool)] = [0.13, 0.59, 0.95]

        for col, panel in enumerate([rgb, gt_vis, pred_vis, err]):
            ax = axes[row][col]
            ax.imshow(panel, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        # IoU / F1 as compact text overlay on the RGB panel (bottom-left corner)
        H = rgb_raw.shape[0]
        axes[row][0].text(
            3, H - 3,
            f"IoU {v['iou']:.2f}  ·  F1 {v['f1']:.2f}",
            fontsize=10, color="white", fontweight="bold", va="bottom",
            bbox=dict(facecolor="black", alpha=0.50, pad=1.5, edgecolor="none"),
        )

    # Compact legend at bottom centre
    fig.legend(
        handles=[mpatches.Patch(facecolor=c, edgecolor="none", label=l)
                 for c, l in LEGEND_ITEMS],
        loc="lower center", ncol=3, fontsize=12,
        frameon=True, framealpha=0.92, edgecolor="#cccccc",
        bbox_to_anchor=(0.5, 0.0), handlelength=1.0,
        handletextpad=0.4, columnspacing=1.2,
    )
    _savefig(fig, path, dpi=130)


# ── Convenience wrapper ───────────────────────────────────────────────────────

def save_all_plots(rows, visuals, out_dir, model_label: str,
                   n_visual: int = 12, iou_key: str = "iou_field",
                   fixed_indices=None):
    """Generate all six standard plots and save to out_dir.

    fixed_indices: pass SHARED_TILE_INDICES so all models show the same tiles.
    """
    out_dir = Path(out_dir)
    print("\nGenerating plots ...")
    plot_summary_bar(rows,              out_dir / "metrics_summary.png",  model_label, iou_key)
    plot_iou_histogram(rows,            out_dir / "iou_histogram.png",    iou_key)
    plot_metrics_boxplot(rows,          out_dir / "metrics_boxplot.png",  iou_key)
    plot_pr_scatter(rows,               out_dir / "pr_scatter.png",       iou_key)
    plot_confusion_matrix(visuals,      out_dir / "confusion_matrix.png")
    plot_visual_grid(visuals, n_visual, out_dir / "visual_grid.png",      fixed_indices)
    print(f"All plots → {out_dir}/")
