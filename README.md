# Field Boundary Delineation using Deep Learning 

Automated delineation of agricultural field boundaries from Sentinel-2 satellite imagery using five model families. All models are benchmarked on the same test set so results are directly comparable.

---

## What this project does

Given a Sentinel-2 satellite image tile, each model produces a pixel-level map of agricultural field boundaries. The benchmark covers the full spectrum from fully supervised training to zero-shot deployment, allowing a direct comparison of how much domain adaptation each approach requires.

---

## Models

| Model | Approach | Training required |
|---|---|---|
| **UNet (8-band)** | Semantic segmentation — 3 classes | Full training on AI4Boundaries |
| **UNet (RGB)** | Same as above, RGB input only | Full training on AI4Boundaries |
| **FTW Pretrained** | UNet, zero-shot cross-dataset transfer | None — download pretrained weights |
| **SAM1 Frozen** | ViT-Base automatic mask generator | None — zero-shot |
| **SAM2 Frozen** | Hiera temporal memory, zero-shot | None — zero-shot |
| **SAM2 Fine-tuned** | Hiera, decoder fine-tuned | Decoder only (~4M params) |
| **DelineateAnything** | YOLO11x instance segmentation | None — zero-shot |


---

## Dataset — AI4Boundaries

The project is built around the **AI4Boundaries** dataset published by the European Joint Research Centre (JRC).

**Paper:** https://essd.copernicus.org/articles/15/317/2023/

### Coverage

- 7 EU countries: Austria, France, Spain (Catalonia), Luxembourg, Netherlands, Slovenia, Sweden
- 7,831 tiles of 256 × 256 pixels at 10 m/pixel (≈ 2.56 × 2.56 km per tile)
- ~2.5 million agricultural parcels across ~47,105 km²
- Ground truth: official 2019 GSAA (EU farm subsidy application) parcel records

### Data splits

```
train/   70%
val/     15%
test/    15%
```

### File structure per tile

```
<split>/sentinel2/images/<id>_S2_10m_256.nc     ← Sentinel-2 image (NetCDF)
<split>/sentinel2/masks/<id>_S2label_10m_256.tif ← Ground truth mask (GeoTIFF)
```

### Sentinel-2 image format

- **Format:** NetCDF with variables B2, B3, B4, B8, NDVI
- **Time axis:** 6 monthly composites (March through August 2019)
- **Resolution:** 10 m/pixel, cloud-free Level-2A surface reflectance
- **Value range:** raw DN / 10000 → approximately [0, 0.5] as float

This project uses two time steps:
- `_TIME_A = 0` → March (bare soil / early crop)
- `_TIME_B = 5` → August (peak vegetation)

### Ground truth mask format

4-band GeoTIFF per tile:

| Band | Content | Values |
|---|---|---|
| 1 | Field extent | 0 = not field, 1 = field |
| 2 | Boundary | 0 = interior, 1 = boundary (1-pixel dilated) |
| 3 | Distance map | 0 at boundary → 1 at field center |
| 4 | Instance ID | Unique integer per field polygon |

The dataset loaders derive a 3-class training mask:
- `0` = background (not field)
- `1` = field interior (band 1 AND NOT band 2)
- `2` = boundary (band 2)

---

## Installation

### 1. Clone and set up environment

```bash
git clone git@github.com:amanchry/field-delineation-models.git
cd field-delineation-models

conda env create -f environment.yml
conda activate fielddl
```

### 2. Download AI4Boundaries data

Open and run `download_ai4boundaries_data.ipynb`. This fetches tiles for your chosen country/region and organises them into the expected folder structure.


## Model details

### UNet (from scratch)

**Architecture:** UNet with EfficientNet-B5 encoder from `segmentation-models-pytorch`.

**Input:** 8-band tensor `[B, 8, 256, 256]` — B4/B3/B2/B8 for March concatenated with B4/B3/B2/B8 for August. An RGB-only variant uses 6 bands (no NIR).

**Output:** 3-class logits `[B, 3, 256, 256]` — background / field interior / boundary.

**Loss function:**
```
Loss = CrossEntropy(weights=[0.3, 1.0, 5.0])        # per-pixel class loss
     + 0.2 × Dice(background) + 0.4 × Dice(field) + 1.4 × Dice(boundary)
```
Boundary is weighted highest because it covers only ~5% of pixels.

**Training:**
```bash
# Edit DATA_ROOT in unet/train.py, then:
python3 unet/train.py

# RGB-only variant:
python3 unet/train_rgb.py
```

---

### FTW Pretrained UNet (zero-shot)

**What it is:** Official baseline models from the [Fields of the World](https://github.com/fieldsoftheworld/ftw-baselines) benchmark, trained on 24 countries with ~70,000 labeled fields. Same UNet + EfficientNet architecture, downloaded as a PyTorch Lightning checkpoint.

**Available models** (set `PRETRAINED_MODEL` in `unet_pretrained/evaluate.py`):

| Key | Backbone | Notes |
|---|---|---|
| `FTW_PRUE_EFNET_B5` | EfficientNet-B5 | Recommended — best accuracy |
| `FTW_PRUE_EFNET_B3` | EfficientNet-B3 | Faster, slightly less accurate |
| `FTW_PRUE_EFNET_B7` | EfficientNet-B7 | Most accurate, slowest |
| `FTW_PRUE_EFNET_B5_CCBY` | EfficientNet-B5 | CC-BY license only training data |

**No training needed — just evaluate:**
```bash
python3 unet_pretrained/evaluate.py
```
Weights are downloaded automatically from the FTW GitHub release on first run.

---

### SAM1 — Segment Anything v1

**Architecture:** ViT-Base image encoder (86M params, frozen) + prompt encoder (frozen) + mask decoder (~4M params, trainable).

**How it works:**
- `SamAutomaticMaskGenerator` sweeps a grid of point prompts across the tile
- For each point, the decoder predicts 3 candidate masks and selects the best
- All accepted masks (predicted_iou ≥ threshold) are merged into a binary field map

**Input:** Single August frame, RGB channels (B4/B3/B2) as uint8 [H, W, 3].

**Zero-shot evaluation (no training):**
```bash
python3 sam1/evaluate_frozen.py
```

**Fine-tune the decoder:**
```bash
# Edit DATA_ROOT and VARIANT in sam1/train.py, then:
python3 sam1/train.py
```

**Evaluate fine-tuned decoder:**
```bash
python3 sam1/evaluate.py
```


---

### SAM2 — Segment Anything v2

**Architecture:** Hiera hierarchical backbone + memory-conditioned mask decoder. Supports temporal context: one frame is encoded into a memory bank before predicting on the second frame.

**Variants:** `tiny` | `small` | `base_plus` | `large` (set `VARIANT` in train/evaluate scripts).

**How it works:**
- March frame → `model.forward_image()` → stored in memory bank
- August frame → `generator.generate()` conditioned on March memory
- Decoder cross-attends to March embeddings when predicting August masks
- Seasonal contrast (bare soil in March vs crop in August) helps locate boundaries

**Zero-shot evaluation:**
```bash
python3 sam2/evaluate_frozen.py
```

**Fine-tune the decoder:**
```bash
# Edit DATA_ROOT and VARIANT in sam2/train.py, then:
python3 sam2/train.py
```

Only the mask decoder is trained. The image encoder and prompt encoder remain frozen.

**Evaluate fine-tuned decoder:**
```bash
python3 sam2/evaluate.py
```

**Fine-tuning settings** (per-variant presets in `sam2/train.py`):

| Variant | Batch size | LR | Grad accum | Effective batch |
|---|---|---|---|---|
| tiny | 16 | 1e-4 | ×1 | 16 |
| small | 8 | 1e-4 | ×2 | 16 |
| base_plus | 4 | 5e-5 | ×4 | 16 |
| large | 2 | 3e-5 | ×8 | 16 |


---

### DelineateAnything (zero-shot)

**Architecture:** YOLO11x instance segmentation model (62M params) pretrained on FBIS-22M — a dataset of 22 million field boundary instances from global satellite imagery.

**Key difference from other models:** produces individual **instance masks** (one per detected field), not a single merged binary map. Each instance has its own mask, confidence score, and bounding box.

**Input:** Single August RGB frame, per-band 1st–99th percentile normalized to [0, 1].

**No training needed:**
```bash
python3 delineateanything/evaluate.py
```


---

## Evaluation metrics explained

### Pixel IoU (per class)

```
IoU(class c) = (predicted c AND ground truth c) / (predicted c OR ground truth c)
```

Measures how well the model covers the area of each class. Reported separately for background, field, and boundary. `mIoU` is the mean across all classes.

### Object-level Precision / Recall / F1

Treats each connected predicted region as one predicted field instance, and each connected GT region as one GT field. Matches them greedily at IoU ≥ 0.5.

```
Precision = matched predictions / total predictions   (how many predictions are real fields)
Recall    = matched predictions / total GT fields     (how many actual fields were found)
F1        = 2 × Precision × Recall / (Precision + Recall)
```

**Why both metrics matter:**

Pixel IoU and Object F1 often tell different stories. A model that predicts one large blob covering all fields gets high pixel IoU but F1 ≈ 0 because no single blob cleanly matches one GT instance. Object F1 is the more meaningful metric for practical field mapping where individual parcel identification matters.

---

## Citation

AI4Boundaries data:
```
Dandrifosse, S. et al. (2023). AI4Boundaries: an open AI-ready dataset to map
field boundaries with Sentinel-2 and aerial photography.
Earth System Science Data, 15, 317–329.
https://doi.org/10.5194/essd-15-317-2023
```

FTW pretrained weights:
```
Fields of the World (FTW) — https://github.com/fieldsoftheworld/ftw-baselines
```

DelineateAnything:
```
DelineateAnything — https://github.com/fieldsoftheworld/delineate-anything
Pretrained on FBIS-22M via torchgeo/delineate-anything (HuggingFace)
```
