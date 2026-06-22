"""
SAM1 (Segment Anything) model builder.

Tries local checkpoint first; falls back to downloading from Meta's servers.

Install the package with:
    pip install git+https://github.com/facebookresearch/segment-anything.git

Note: SAM1 image encoder requires float32. Apple MPS does not support float64
so the automatic mask generator must run on CPU when MPS is the device.
"""

import urllib.request
from pathlib import Path

_CKPT_NAMES = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}

_CKPT_URLS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}

# Weights live in sam1/weights/ (next to this file)
_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


def build_sam1(
    variant: str = "vit_b",
    checkpoint: str = None,
    device: str = "cuda",
    mode: str = "train",
):
    """
    Build and return a SAM1 model.

    variant    : vit_b | vit_l | vit_h
    checkpoint : path to .pth file; None = auto from weights/sam1/
    device     : cuda | mps | cpu
    mode       : 'train' (freeze encoder+prompt, train decoder) | 'eval'
    """
    try:
        from segment_anything import sam_model_registry
    except ImportError:
        raise ImportError(
            "segment-anything package required.\n"
            "Install: pip install git+https://github.com/facebookresearch/segment-anything.git"
        )

    if variant not in _CKPT_NAMES:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(_CKPT_NAMES)}")

    if checkpoint is None:
        local = _WEIGHTS_DIR / _CKPT_NAMES[variant]
        checkpoint = str(local) if local.exists() else None

    if checkpoint is None or not Path(checkpoint).exists():
        local = _WEIGHTS_DIR / _CKPT_NAMES[variant]
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading SAM1-{variant} weights ...")
        urllib.request.urlretrieve(_CKPT_URLS[variant], str(local))
        checkpoint = str(local)
        print(f"  Saved → {local}")

    model = sam_model_registry[variant](checkpoint=checkpoint)
    model.to(device)
    print(f"Loaded SAM1-{variant} from {checkpoint}")

    if mode == "train":
        for p in model.image_encoder.parameters():
            p.requires_grad = False
        model.image_encoder.eval()
        for p in model.prompt_encoder.parameters():
            p.requires_grad = False
        model.prompt_encoder.eval()
        for p in model.mask_decoder.parameters():
            p.requires_grad = True
        model.mask_decoder.train()
    else:
        model.eval()

    return model
