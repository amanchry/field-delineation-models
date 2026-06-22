"""
AI4Boundaries dataset loader.

Layout expected:
  <root>/<split>/sentinel2/images/<id>_S2_10m_256.nc
  <root>/<split>/sentinel2/masks/<id>_S2label_10m_256.tif

Image: NetCDF with 5 bands × 6 monthly composites (B2, B3, B4, B8, NDVI).
Mask : GeoTIFF band 1 = field extent (0/1), band 2 = boundary (0/1).

Output sample (AI4BoundariesDataset):
    num_channels=4 (default) → image [8, 256, 256]  float32  B4/B3/B2/B8 × March+August
    num_channels=3 (RGB only) → image [6, 256, 256]  float32  B4/B3/B2    × March+August
    mask  : Tensor [256, 256]     int64    (0=background, 1=field, 2=boundary)
"""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import rasterio
import torch
import xarray as xr
from scipy.ndimage import label as cc_label
from torch.utils.data import Dataset

SCALE = 10000.0

# AI4Boundaries NetCDF variable order: B2, B3, B4, B8, NDVI
# Reorder to B4, B3, B2, B8 to match the standard FTW channel convention
_BANDS    = ["B4", "B3", "B2", "B8"]
_TIME_A   = 0   # March  (spring)
_TIME_B   = 5   # August (autumn)


def _collect_samples(root: Path, split: str) -> list:
    img_dir  = root / split / "sentinel2" / "images"
    mask_dir = root / split / "sentinel2" / "masks"
    if not img_dir.exists():
        raise FileNotFoundError(f"Split not found: {img_dir}")
    samples = []
    for nc in sorted(img_dir.glob("*_S2_10m_256.nc")):
        sid  = nc.stem.replace("_S2_10m_256", "")
        mask = mask_dir / f"{sid}_S2label_10m_256.tif"
        if mask.exists():
            samples.append({"image": str(nc), "mask": str(mask)})
    if not samples:
        raise RuntimeError(f"No samples in {img_dir}")
    return samples


class AI4BoundariesDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms: Optional[Callable] = None,
        num_channels: int = 4,   # 4 = B4/B3/B2/B8 (RGB+NIR), 3 = B4/B3/B2 (RGB only)
    ):
        self.transforms   = transforms
        self.samples      = _collect_samples(Path(root), split)
        self.bands_sel    = _BANDS[:num_channels]   # e.g. ["B4","B3","B2"] for num_channels=3

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        p  = self.samples[idx]
        ds = xr.open_dataset(p["image"])
        bands = []
        for t in (_TIME_A, _TIME_B):
            for b in self.bands_sel:
                bands.append(ds[b].values[t])
        ds.close()
        # num_channels=4 → [8,256,256]  |  num_channels=3 → [6,256,256]
        image = np.stack(bands, axis=0).astype(np.float32) / SCALE

        with rasterio.open(p["mask"]) as f:
            extent   = f.read(1).astype(np.uint8)
            boundary = f.read(2).astype(np.uint8)

        mask = np.zeros_like(extent, dtype=np.int64)
        mask[extent == 1]   = 1
        mask[boundary == 1] = 2

        sample = {
            "image": torch.from_numpy(image),
            "mask":  torch.from_numpy(mask),
        }
        if self.transforms:
            sample = self.transforms(sample)
        return sample


class AI4BoundariesSAM1Dataset(Dataset):
    """
    AI4Boundaries dataset for SAM1 fine-tuning.

    SAM1 is single-frame — uses only the August (t=5) composite as input.
    Instance mask comes from Band 4 (Field ID) of the mask GeoTIFF.

    Sample dict:
        image : Tensor [C, 256, 256]  float32  (normalised 0-1)
        mask  : Tensor [256, 256]     int64    (instance IDs; 0=background)
    """

    _TIME_B = 5   # August

    def __init__(
        self,
        root: str,
        split: str = "train",
        num_channels: int = 3,
        transforms=None,
    ):
        self.num_channels = num_channels   # 3 = RGB only (SAM1 encoder is 3-ch)
        self.transforms   = transforms
        self.samples      = _collect_samples(Path(root), split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        p = self.samples[idx]
        bands_sel = _BANDS[: self.num_channels]

        ds = xr.open_dataset(p["image"])
        win_b = [ds[b].values[self._TIME_B] for b in bands_sel]
        ds.close()

        image = np.stack(win_b, axis=0).astype(np.float32) / SCALE

        with rasterio.open(p["mask"]) as f:
            field_id = f.read(4).astype(np.int64)
        field_id[field_id < 0] = 0

        sample = {
            "image": torch.from_numpy(image),
            "mask":  torch.from_numpy(field_id),
        }
        if self.transforms:
            sample = self.transforms(sample)
        return sample


class AI4BoundariesSAM2Dataset(Dataset):
    """
    AI4Boundaries dataset for SAM2 fine-tuning.

    Returns March (t=0) as temporal context (window_a) and August (t=5) as
    the prediction frame (window_b). Instance mask uses Band 4 (Field ID).

    Sample dict:
        window_a : Tensor [C, 256, 256]  float32
        window_b : Tensor [C, 256, 256]  float32
        mask     : Tensor [256, 256]     int64    (instance IDs; 0=background)
    """

    _TIME_A = 0
    _TIME_B = 5

    def __init__(
        self,
        root: str,
        split: str = "train",
        num_channels: int = 3,
        transforms: Optional[Callable] = None,
    ):
        self.num_channels = num_channels
        self.transforms = transforms
        self.samples = _collect_samples(Path(root), split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        p = self.samples[idx]
        bands_sel = _BANDS[: self.num_channels]

        ds = xr.open_dataset(p["image"])
        win_a, win_b = [], []
        for band in bands_sel:
            win_a.append(ds[band].values[self._TIME_A])
            win_b.append(ds[band].values[self._TIME_B])
        ds.close()

        window_a = np.stack(win_a, axis=0).astype(np.float32) / SCALE
        window_b = np.stack(win_b, axis=0).astype(np.float32) / SCALE

        with rasterio.open(p["mask"]) as f:
            field_id = f.read(4).astype(np.int64)
        field_id[field_id < 0] = 0

        sample = {
            "window_a": torch.from_numpy(window_a),
            "window_b": torch.from_numpy(window_b),
            "mask":     torch.from_numpy(field_id),
        }
        if self.transforms:
            sample = self.transforms(sample)
        return sample
