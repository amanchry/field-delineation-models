"""
SAM2 model builder.

Weights are stored in sam2/weights/ (next to this file).
Tries local checkpoint first; falls back to downloading from Hugging Face.

Install sam2 with:
    pip install git+https://github.com/facebookresearch/sam2.git
"""

import site
import sys
from pathlib import Path

VARIANT_MAP = {
    "tiny":      ("configs/sam2.1/sam2.1_hiera_t.yaml",  "facebook/sam2.1-hiera-tiny"),
    "small":     ("configs/sam2.1/sam2.1_hiera_s.yaml",  "facebook/sam2.1-hiera-small"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "facebook/sam2.1-hiera-base-plus"),
    "large":     ("configs/sam2.1/sam2.1_hiera_l.yaml",  "facebook/sam2.1-hiera-large"),
}

_CKPT_NAMES = {
    "tiny":      "sam2.1_hiera_tiny.pt",
    "small":     "sam2.1_hiera_small.pt",
    "base_plus": "sam2.1_hiera_base_plus.pt",
    "large":     "sam2.1_hiera_large.pt",
}

# Weights live in sam2/weights/ (next to this file)
_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

# ── Installed-package helpers ─────────────────────────────────────────────────

def _installed_sam2_site() -> str:
    """Return the site-packages directory that contains the installed sam2 library."""
    local_dir = Path(__file__).resolve().parent
    candidates = site.getsitepackages()
    try:
        candidates = candidates + [site.getusersitepackages()]
    except Exception:
        pass
    for sp in candidates:
        sam2_pkg = Path(sp) / "sam2"
        if sam2_pkg.exists() and sam2_pkg.resolve() != local_dir:
            return sp
    raise ImportError(
        "Installed sam2 package not found.\n"
        "Install: pip install git+https://github.com/facebookresearch/sam2.git"
    )


def _with_installed_sam2(fn):
    """
    Call fn() with sys.path temporarily set so the installed sam2 package is
    visible instead of the local sam2/ folder.

    The local sam2/ folder shadows the installed package because:
      - sys.path contains the project root  (field-delineation-models/)
      - sys.path also contains ''           (= cwd after os.chdir(ROOT))
    Both resolve to the project root, which has a sam2/ subfolder.
    We remove those two entries and prepend the real site-packages directory.
    """
    sp = _installed_sam2_site()
    project_root = str(Path(__file__).resolve().parent.parent)

    saved_path = sys.path.copy()
    sys.path = [sp] + [
        p for p in sys.path
        if p not in ('', '.', project_root)
        and Path(p).resolve() != Path(project_root)
    ]

    # Clear any cached local-package entries so Python re-imports from site-packages
    evicted = {k: sys.modules.pop(k) for k in list(sys.modules)
               if k == "sam2" or k.startswith("sam2.")}
    try:
        return fn()
    finally:
        sys.path = saved_path
        for k, v in evicted.items():
            sys.modules.setdefault(k, v)


# ── Public API ────────────────────────────────────────────────────────────────

def build_sam2(
    variant: str = "small",
    checkpoint: str = None,
    device: str = "cuda",
    mode: str = "train",
):
    """
    Build and return a SAM2 model.

    variant    : tiny | small | base_plus | large
    checkpoint : path to .pt file; None = auto from sam2/weights/
    device     : cuda | mps | cpu
    mode       : 'train' (freeze encoder+prompt, train decoder) | 'eval'
    """
    if variant not in VARIANT_MAP:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(VARIANT_MAP)}")

    config_yaml, hf_id = VARIANT_MAP[variant]

    if checkpoint is None:
        local = _WEIGHTS_DIR / _CKPT_NAMES[variant]
        checkpoint = str(local) if local.exists() else None

    if checkpoint is None or not Path(checkpoint).exists():
        local = _WEIGHTS_DIR / _CKPT_NAMES[variant]
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading SAM2-{variant} from Hugging Face ...")
        def _download():
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id=hf_id, filename=_CKPT_NAMES[variant], local_dir=str(local.parent))
        _with_installed_sam2(_download)
        checkpoint = str(local)
        print(f"  Saved → {local}")

    def _load():
        from sam2.build_sam import build_sam2 as _build
        return _build(config_yaml, checkpoint, device=device)

    model = _with_installed_sam2(_load)
    print(f"Loaded SAM2-{variant} from {checkpoint}")

    if mode == "train":
        for p in model.image_encoder.parameters():
            p.requires_grad = False
        model.image_encoder.eval()
        for p in model.sam_prompt_encoder.parameters():
            p.requires_grad = False
        model.sam_prompt_encoder.eval()
        for p in model.sam_mask_decoder.parameters():
            p.requires_grad = True
        model.sam_mask_decoder.train()
    else:
        model.eval()

    return model


def load_sam2_mask_generator(model):
    """Return a SAM2AutomaticMaskGenerator using the installed sam2 library."""
    def _make():
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        return SAM2AutomaticMaskGenerator(model)
    return _with_installed_sam2(_make)
