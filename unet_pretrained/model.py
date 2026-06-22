"""
Load a pretrained FTW UNet checkpoint for zero-shot evaluation.

Available model keys — see model_registry.py for the full list.
Recommended: FTW_PRUE_EFNET_B5 or FTW_PRUE_EFNET_B5_CCBY
"""

from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from unet_pretrained.model_registry import MODEL_REGISTRY


def load_model_from_checkpoint(path: str, ckpt: dict = None) -> tuple[nn.Module, str]:
    """Load a UNet model from a Lightning .ckpt checkpoint file."""
    if ckpt is None:
        ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    hparams   = ckpt["hyper_parameters"]
    model_type = hparams["model"]
    state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}

    for key in ("criterion.weight", "ce_loss.weight"):
        state_dict.pop(key, None)

    if model_type in ("unet", "unet_r"):
        kwargs = {}
        if model_type == "unet_r":
            kwargs["decoder_channels"] = (16, 32, 64, 128, 256)
        model = smp.Unet(
            encoder_name=hparams["backbone"],
            encoder_weights=None,
            in_channels=hparams["in_channels"],
            classes=hparams["num_classes"],
            **kwargs,
        )
    elif model_type == "upernet":
        model = smp.UPerNet(
            encoder_name=hparams["backbone"],
            encoder_weights=None,
            in_channels=hparams["in_channels"],
            classes=hparams["num_classes"],
        )
    elif model_type == "deeplabv3+":
        model = smp.DeepLabV3Plus(
            encoder_name=hparams["backbone"],
            encoder_weights=None,
            in_channels=hparams["in_channels"],
            classes=hparams["num_classes"],
        )
    else:
        raise ValueError(f"Unsupported model type in checkpoint: {model_type}")

    model.load_state_dict(state_dict, strict=True)
    return model, model_type


def load_pretrained_model(model_name: str, weights_dir: str, device: str) -> tuple[nn.Module, int]:
    """
    Download (if needed) and load a pretrained FTW model by registry name.

    Returns the model in eval mode, ready for patchwise inference.

    Args:
        model_name:  Registry name, e.g. "FTW_PRUE_EFNET_B5".
        weights_dir: Local directory to cache downloaded .ckpt files.
        device:      'cuda', 'mps', or 'cpu'.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown pretrained model '{model_name}'.\n"
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    spec      = MODEL_REGISTRY[model_name]
    cache_dir = Path(weights_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(cache_dir / f"{model_name}.ckpt")

    if not Path(ckpt_path).exists():
        print(f"  Downloading {model_name} ...")
        print(f"  Source : {spec.url}")
        print(f"  Saving → {ckpt_path}")
        torch.hub.download_url_to_file(spec.url, ckpt_path, progress=True)
    else:
        print(f"  Using cached checkpoint: {ckpt_path}")

    ckpt        = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    num_classes = ckpt["hyper_parameters"]["num_classes"]
    model, model_type = load_model_from_checkpoint(ckpt_path, ckpt=ckpt)
    model = model.eval().to(device)
    print(f"  Loaded {model_name}  (arch={model_type}, classes={num_classes})")
    return model, num_classes
