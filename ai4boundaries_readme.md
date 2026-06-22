# AI4Boundaries Dataset

**Paper:** [AI4Boundaries: an open AI-ready dataset to map field boundaries with Sentinel-2 and aerial photography](https://essd.copernicus.org/articles/15/317/2023/)
**GitHub:** https://github.com/waldnerf/ai4boundaries
**Data:** https://data.jrc.ec.europa.eu/dataset/0e79ce5d-e4c8-4721-8773-59a4acf2c9c9

Availbale data for Austria, Catalonia, France, Luxembourg, the Netherlands, Slovenia, and Sweden

---

## What is AI4Boundaries?

AI4Boundaries is a benchmark dataset for agricultural field boundary delineation across 7 European countries. It pairs satellite imagery (Sentinel-2) and high-resolution aerial orthophotos with ground-truth parcel boundaries derived from official 2019 GSAA (Geospatial Aid Application) records — the same data farmers submit to claim EU agricultural subsidies.

**Coverage:** Austria · Catalonia (Spain) · France · Luxembourg · Netherlands · Slovenia · Sweden
**Total samples:** 7,831 tiles of 4×4 km, containing ~2.5 million parcels over ~47,105 km²

---

## Top-Level Files

| File | Description |
|------|-------------|
| `ai4boundaries_ftp_urls_all.csv` | Master index of every file in the dataset. Each row has the FTP URL, country, sample ID, split (train/val/test), and modality. Use this to selectively download subsets. |
| `download_ai4boundaries_france.py` | Script to download the France (FR) subset from the JRC FTP server. |
| `ai4boundaries_FR/` | Downloaded data — France only (this repo). |

> The full dataset also includes a `sampling/` folder with:
> - `ai4boundaries_sampling.gpkg` — 7,831 4×4 km grid polygons with stratification attributes
> - `ai4boundaries_parcels_vector.gpkg` — original vector parcel boundaries

---

## Downloaded Folder Structure

```
ai4boundaries_FR/
├── train/
│   └── sentinel2/
│       ├── images/     ← Sentinel-2 image tiles (NetCDF)
│       └── masks/      ← Corresponding label masks (GeoTIFF)
├── val/
│   └── sentinel2/
│       ├── images/
│       └── masks/
└── test/
    └── sentinel2/
        ├── images/
        └── masks/
```

The orthophoto modality follows the same structure under an `orthophoto/` sibling directory (not yet downloaded).

---

Sentinel-2 — satellite imagery, coarse but rich in time
- 10 m/pixel resolution
- 256×256 px chip = covers 2.56 km × 2.56 km
- 5 spectral bands: R, G, B, NIR, NDVI
- 6 time steps (monthly composites, March–August 2019) — so it's a time series, not a single snapshot
- Globally consistent, free, easy to scale

Orthophoto — aerial photography, fine detail but static
- 1 m/pixel resolution (10× sharper)
- 512×512 px chip = covers only 512 m × 512 m (a much smaller area on the ground)
- 3 bands: R, G, B only (NIR was dropped because not all countries provided it)
- Single date in 2019 — no time series
- Collected by national mapping agencies, so availability varies by country (233 Swedish tiles are missing due to licensing)



## Folder Details

### `sentinel2/images/`

**What:** Sentinel-2 multispectral image chips.
**Format:** NetCDF (`.nc`)
**Naming:** `{COUNTRY}_{sampleID}_S2_10m_256.nc`
**Example:** `FR_54222_S2_10m_256.nc`

| Property | Value |
|----------|-------|
| Spatial resolution | 10 m/pixel |
| Chip size | 256 × 256 pixels (= 2,560 × 2,560 m = 2.56 × 2.56 km) |
| Bands | Red, Green, Blue, NIR, NDVI |
| Temporal | Monthly composites, **March – August 2019** (6 time steps) |
| Projection | EPSG:3035 (LAEA Europe) |

Each `.nc` file contains a time series of cloud-free monthly composites built from Sentinel-2 Level-2A surface reflectance data. The chip represents a 4×4 km tile subsetted to 2.56 km to avoid boundary effects.

---

### `sentinel2/masks/`

**What:** Pixel-level ground-truth labels derived from GSAA parcel polygons, aligned to the Sentinel-2 grid.
**Format:** GeoTIFF (`.tif`), 4 bands
**Naming:** `{COUNTRY}_{sampleID}_S2label_10m_256.tif`
**Example:** `FR_54222_S2label_10m_256.tif`

| Band | Name | Description |
|------|------|-------------|
| 1 | **Extent** | Binary mask — `1` = agricultural field pixel, `0` = background |
| 2 | **Boundary** | Binary mask — `1` = field boundary pixel (dilated parcel edges) |
| 3 | **Distance** | Normalized distance transform — pixel value = distance to nearest boundary (0 at boundary → 1 at field center) |
| 4 | **Field ID** | Integer ID enumerating individual parcel instances within the chip |

The boundary band uses a 1-pixel dilation of the rasterized parcel edges. The distance band enables regression-based boundary detection models. Together these four bands support segmentation, boundary detection, and instance segmentation tasks.

---

### `orthophoto/images/` 

**What:** High-resolution aerial orthophotos from national mapping agencies.
**Format:** GeoTIFF (`.tif`)
**Naming:** `{COUNTRY}_{sampleID}_ortho_1m_512.tif`

| Property | Value |
|----------|-------|
| Spatial resolution | 1 m/pixel |
| Chip size | 512 × 512 pixels (= 512 × 512 m) |
| Bands | Red, Green, Blue (RGB only — NIR dropped for cross-country consistency) |
| Temporal | Single acquisition, 2019 |

Note: 233 samples (mostly Swedish) have no orthophoto due to licensing restrictions. Total available: 7,598 of 7,831 tiles.

---

### `orthophoto/masks/` 

**What:** Same ground-truth labels as Sentinel-2 masks, resampled to 1 m resolution on the orthophoto grid.
**Format:** GeoTIFF (`.tif`), 4 bands
**Naming:** `{COUNTRY}_{sampleID}_ortholabel_1m_512.tif`

Same 4-band structure as the Sentinel-2 masks (Extent · Boundary · Distance · Field ID), matched pixel-for-pixel to the orthophoto chip.

---

## Train / Val / Test Split

Splits are pre-defined and stored in `ai4boundaries_ftp_urls_all.csv`. Stratified random sampling ensures each split covers the full range of field sizes and landscape fragmentation.

| Split | Fraction | ~Tiles (global) |
|-------|----------|-----------------|
| train | 70% | 5,319 |
| val | 15% | 1,140 |
| test | 15% | 1,139 |

---


