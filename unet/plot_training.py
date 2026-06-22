"""
Plot training curves from train_log.csv.

Usage:
    python3 unet/plot_training.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
out_folder = "efficientnet-b5_brittany_rgb"

_OUTPUTS = Path(__file__).parent / "outputs"
LOG_PATH = _OUTPUTS / out_folder / "train_log.csv"
OUT_PATH = _OUTPUTS / out_folder / "training_curve.png"


# ─────────────────────────────────────────────────────────────────────────────

def load(path: Path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        "epoch":      [int(r["epoch"])        for r in rows],
        "train_miou": [float(r["train_miou"]) for r in rows],
        "val_miou":   [float(r["val_miou"])   for r in rows],
        "train_loss": [float(r["train_loss"]) for r in rows],
        "val_loss":   [float(r["val_loss"])   for r in rows],
    }


def smooth(values, window=5):
    kernel = np.ones(window) / window
    pad    = window // 2
    padded = np.pad(values, pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


def main():
    if not LOG_PATH.exists():
        print(f"Log not found: {LOG_PATH}")
        print("Train first:  python3 unet/train.py")
        return

    data   = load(LOG_PATH)
    ep     = data["epoch"]
    best_i = int(np.argmax(data["val_miou"]))
    best_e = ep[best_i]
    best_v = data["val_miou"][best_i]

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(f" UNet — Training Curves", fontsize=13, fontweight="bold")
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ── mIoU panel ────────────────────────────────────────────────────────────
    ax1.plot(ep, data["train_miou"], color="#2196F3", linewidth=1.2,
             linestyle="--", alpha=0.6, label="Train mIoU")
    ax1.plot(ep, data["val_miou"],   color="#4CAF50", linewidth=1.0,
             alpha=0.3, label="_nolegend_")
    ax1.plot(ep, smooth(data["val_miou"]), color="#4CAF50", linewidth=2.2,
             label="Val mIoU (smoothed)")
    ax1.axvline(best_e, color="#4CAF50", linestyle=":", linewidth=1.2, alpha=0.8)
    ax1.scatter([best_e], [best_v], color="#4CAF50", zorder=5, s=70,
                label=f"Best: {best_v:.4f} @ ep {best_e}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("mIoU")
    ax1.set_title("mIoU — Train vs Val")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25, linestyle="--")

    # ── Loss panel ────────────────────────────────────────────────────────────
    ax2.plot(ep, data["train_loss"], color="#2196F3", linewidth=1.2,
             linestyle="--", alpha=0.6, label="Train loss (CE+Dice)")
    ax2.plot(ep, data["val_loss"],   color="#FF9800", linewidth=1.0,
             alpha=0.3, label="_nolegend_")
    ax2.plot(ep, smooth(data["val_loss"]), color="#FF9800", linewidth=2.2,
             label="Val loss (smoothed, CE only)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss — Train vs Val")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25, linestyle="--")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PATH}")
    print(f"Best val mIoU : {best_v:.4f} at epoch {best_e}")
    print(f"Latest train  : {data['train_miou'][-1]:.4f}  val: {data['val_miou'][-1]:.4f}")


if __name__ == "__main__":
    main()
