import os
import torch


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    return "cpu"


def amp_context(device: str):
    if device == "cuda":
        return torch.amp.autocast("cuda")
    if device == "mps":
        return torch.amp.autocast("mps", dtype=torch.bfloat16)
    return torch.amp.autocast("cpu", enabled=False)


def grad_scaler(device: str) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
