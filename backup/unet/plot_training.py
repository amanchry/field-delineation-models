"""
Plot epoch vs val mIoU and loss from train_log.csv.

Usage:
    python3 unet/plot_training.py           # 8-band model
    python3 unet/plot_training.py rgb       # RGB-only model
    python3 unet/plot_training.py both      # overlay both on same axes
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

BACKBONE   = "efficientnet-b5"
_OUTPUTS   = Path(__file__).parent / "outputs"
_MODE      = sys.argv[1] if len(sys.argv) > 1 else "8band"


def load_log(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _plot_single(rows, label, color_miou, color_loss, ax_miou, ax_loss):
    epochs     = [int(r["epoch"])        for r in rows]
    train_miou = [float(r["train_miou"]) for r in rows]
    val_miou   = [float(r["val_miou"])   for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    val_loss   = [float(r["val_loss"])   for r in rows]

    best_epoch = epochs[val_miou.index(max(val_miou))]
    best_miou  = max(val_miou)

    ax_miou.plot(epochs, train_miou, label=f"{label} train", color=color_miou,
                 linewidth=1.5, linestyle="--", alpha=0.7)
    ax_miou.plot(epochs, val_miou,   label=f"{label} val",   color=color_miou, linewidth=2)
    ax_miou.axvline(best_epoch, color=color_miou, linestyle=":", alpha=0.5,
                    label=f"{label} best ep{best_epoch} ({best_miou:.4f})")

    ax_loss.plot(epochs, train_loss, label=f"{label} train", color=color_loss,
                 linewidth=1.5, linestyle="--", alpha=0.7)
    ax_loss.plot(epochs, val_loss,   label=f"{label} val",   color=color_loss, linewidth=2)

    return best_epoch, best_miou


def main():
    log_8band = _OUTPUTS / BACKBONE         / "train_log.csv"
    log_rgb   = _OUTPUTS / f"{BACKBONE}_rgb" / "train_log.csv"

    if _MODE == "both":
        missing = [p for p in (log_8band, log_rgb) if not p.exists()]
        if missing:
            for p in missing:
                print(f"Missing: {p}")
            return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        e1, m1 = _plot_single(load_log(log_8band), "8-band", "#2196F3", "#FF9800", ax1, ax2)
        e2, m2 = _plot_single(load_log(log_rgb),   "RGB",    "#4CAF50", "#E91E63", ax1, ax2)
        ax1.set_title(f"UNet {BACKBONE} — mIoU comparison")
        ax2.set_title(f"UNet {BACKBONE} — Loss comparison")
        out_path = _OUTPUTS / "training_curve_both.png"
        print(f"8-band best val mIoU : {m1:.4f} at epoch {e1}")
        print(f"RGB    best val mIoU : {m2:.4f} at epoch {e2}")

    elif _MODE == "rgb":
        if not log_rgb.exists():
            print(f"No log found at {log_rgb}. Train first:  python3 unet/train_rgb.py")
            return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        e, m = _plot_single(load_log(log_rgb), "RGB", "#4CAF50", "#E91E63", ax1, ax2)
        ax1.set_title(f"UNet-RGB ({BACKBONE}) — mIoU")
        ax2.set_title(f"UNet-RGB ({BACKBONE}) — Loss")
        out_path = log_rgb.parent / "training_curve.png"
        print(f"Best val mIoU: {m:.4f} at epoch {e}")

    else:  # 8band (default)
        if not log_8band.exists():
            print(f"No log found at {log_8band}. Train first:  python3 unet/train.py")
            return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        e, m = _plot_single(load_log(log_8band), "8-band", "#2196F3", "#FF9800", ax1, ax2)
        ax1.set_title(f"UNet ({BACKBONE}) — mIoU")
        ax2.set_title(f"UNet ({BACKBONE}) — Loss")
        out_path = log_8band.parent / "training_curve.png"
        print(f"Best val mIoU: {m:.4f} at epoch {e}")

    for ax in (ax1, ax2):
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    ax1.set_ylabel("mIoU")
    ax2.set_ylabel("Loss")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
