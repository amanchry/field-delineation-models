"""
Vectorize binary prediction masks → GeoPackage.

Usage in any evaluate.py:

    from shared.vectorize import masks_to_gpkg

    # dataset.samples  — list of {"image": ..., "mask": str(path_to_geotiff)}
    # pred_masks       — list of np.ndarray [H, W] uint8 binary (1=field, 0=background)

    masks_to_gpkg(dataset.samples, pred_masks, out_dir / "predicted_fields.gpkg")

Output GeoPackage layer "fields" columns:
    geometry    — polygon/multipolygon in the source CRS (EPSG:3035)
    sample_idx  — position in the test split (0-based)
    tile        — stem of the mask filename (chip ID)
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape


def masks_to_gpkg(
    samples: list,
    pred_masks: list,
    out_path,
    layer: str = "fields",
    min_pixels: int = 10,
) -> None:
    """
    Convert a list of binary prediction masks to a single merged GeoPackage.

    Parameters
    ----------
    samples     : dataset.samples list — each entry must have a "mask" key pointing
                  to a GeoTIFF whose CRS and transform are used for georeferencing.
    pred_masks  : parallel list of np.ndarray [H, W], values 0/1 (or bool).
    out_path    : destination .gpkg path.
    layer       : GeoPackage layer name.
    min_pixels  : polygons smaller than this many pixels are dropped (noise filter).
    """
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError("geopandas required:  pip install geopandas")

    from shapely.geometry import shape as _shape

    records = []
    crs = None

    for idx, (sample, pred) in enumerate(zip(samples, pred_masks)):
        mask_path = sample["mask"]

        with rasterio.open(mask_path) as src:
            transform = src.transform
            if crs is None:
                crs = src.crs

        pred_u8 = pred.astype(np.uint8)

        for geom_dict, val in shapes(pred_u8, transform=transform):
            if val != 1:
                continue
            geom = _shape(geom_dict)
            if geom.is_empty:
                continue
            # rough pixel-count filter using area / pixel_area
            pixel_area = abs(transform.a * transform.e)
            if geom.area < min_pixels * pixel_area:
                continue
            records.append({
                "geometry":   geom,
                "sample_idx": idx,
                "tile":       Path(mask_path).stem,
            })

    if not records:
        print(f"  [vectorize] No field polygons to write → {out_path}")
        return

    gdf = gpd.GeoDataFrame(records, crs=crs)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(str(out_path), driver="GPKG", layer=layer)
    print(f"  GeoPackage ({len(gdf)} polygons, {gdf['tile'].nunique()} tiles) → {out_path}")
