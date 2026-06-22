"""
DelineateAnything model loader.

YOLO-based instance segmentation model pretrained on the FBIS-22M dataset.
Single RGB frame input — no temporal context, no training required.

Weights are downloaded from HuggingFace and cached in delineateanything/weights/.
"""

import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms.v2 as T

# Weights are stored next to this file
_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


_CHECKPOINTS = {
    "DelineateAnything-S": (
        "https://huggingface.co/MykolaL/DelineateAnything/resolve/main/DelineateAnything-S.pt"
    ),
    "DelineateAnything": (
        "https://huggingface.co/MykolaL/DelineateAnything/resolve/main/DelineateAnything.pt"
    ),
}


def _ensure_weights(model_name: str) -> Path:
    url      = _CHECKPOINTS[model_name]
    filename = url.split("/")[-1]
    dest     = _WEIGHTS_DIR / filename
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {model_name} weights ...")
        urllib.request.urlretrieve(url, str(dest))
        print(f"  Saved → {dest}")
    return dest


class DelineateAnythingModel:
    """
    Wraps a YOLO DelineateAnything checkpoint for patch-level inference.

    Input : RGB float32 tensor [C, H, W] in [0, 1] after per-band percentile normalisation
    Output: list of ultralytics Results objects
    """

    def __init__(
        self,
        model_name: str = "DelineateAnything-S",
        patch_size: int = 256,
        resize_factor: int = 2,
        conf_threshold: float = 0.05,
        iou_threshold: float = 0.3,
        max_detections: int = 300,
        device: str = "cpu",
    ):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics required: pip install ultralytics")

        if model_name not in _CHECKPOINTS:
            raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(_CHECKPOINTS)}")

        ckpt_path = _ensure_weights(model_name)

        self.patch_size    = (patch_size, patch_size)
        self.image_size    = (patch_size * resize_factor, patch_size * resize_factor)
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.max_detections = max_detections
        self.device         = device
        self.model_name     = model_name

        self.yolo = YOLO(str(ckpt_path)).to(device)
        self.yolo.eval()
        self.yolo.fuse()

        # S2 float [0, 1] → clipped [0, 1] float32 resized to model input size
        self.transforms = nn.Sequential(
            T.Lambda(lambda x: x.unsqueeze(0) if x.ndim == 3 else x),
            T.Lambda(lambda x: x[:, :3]),
            T.Lambda(lambda x: x.clip(0.0, 1.0)),
            T.Resize(self.image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ConvertImageDtype(torch.float32),
        ).to(device)

        print(f"Loaded {model_name} from {ckpt_path}")

    def __call__(self, image: torch.Tensor):
        """
        image : [C, H, W] or [B, C, H, W] float32, values in [0, 1].
        Returns list of ultralytics Results.
        """
        img = self.transforms(image.to(self.device))
        results = self.yolo.predict(
            img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            device=self.device,
            half=False,
            verbose=False,
        )
        for r in results:
            if r.masks is not None:
                r.masks.orig_shape = self.patch_size
            if r.boxes is not None:
                r.boxes.orig_shape = self.patch_size
        return results
