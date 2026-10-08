"""
One place that picks the compute device, so the same code runs on Colab
(NVIDIA CUDA), this laptop (Intel Arc iGPU via PyTorch XPU) and plain CPU.

    CIVITAS_DEVICE=cpu|cuda|xpu   forces a device (default: best available)

Order: cuda > xpu > cpu. Mixed precision:
    cuda Ampere+ -> bf16    cuda older (T4) -> fp16 + GradScaler
    xpu          -> bf16    cpu             -> none (fp32)
"""

import os

import torch


def xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def best_device() -> torch.device:
    forced = os.getenv("CIVITAS_DEVICE", "").strip().lower()
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if xpu_available():
        return torch.device("xpu")
    return torch.device("cpu")


def resolve(device="auto") -> torch.device:
    return best_device() if device in (None, "auto") else torch.device(device)


def supports_bf16(device: torch.device) -> bool:
    if device.type == "cuda":
        return torch.cuda.get_device_capability(device)[0] >= 8
    return device.type == "xpu"


def amp_dtype(device: torch.device, disable: bool = False):
    """(autocast dtype or None, needs GradScaler)."""
    if disable or device.type == "cpu":
        return None, False
    if supports_bf16(device):
        return torch.bfloat16, False
    return torch.float16, True


def half_dtype(device: torch.device):
    """Weight dtype for inference: fp16 on CUDA, bf16 on XPU, fp32 on CPU."""
    if device.type == "cuda":
        return torch.float16
    if device.type == "xpu":
        return torch.bfloat16
    return torch.float32


def manual_seed_all(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if xpu_available():
        torch.xpu.manual_seed_all(seed)


def describe() -> dict:
    d = best_device()
    info = {"device": d.type}
    if d.type == "cuda":
        info.update(name=torch.cuda.get_device_name(0),
                    mem_gb=round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
                    peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
    elif d.type == "xpu":
        info.update(name=torch.xpu.get_device_name(0),
                    mem_gb=round(torch.xpu.get_device_properties(0).total_memory / 1e9, 1),
                    peak_gb=round(torch.xpu.max_memory_allocated() / 1e9, 2))
    return info
