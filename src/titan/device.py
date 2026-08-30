"""Device, precision and throughput setup, kept in one place.

Nothing else in the repo is allowed to hardcode a device or a dtype. The rule
is: pick the device once, set the CUDA backend flags once, and hand the rest of
the code an autocast context it can wrap a forward pass in. Metrics stay out of
this -- ADE/FDE are reported numbers and are always computed in fp32.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Precision:
    enabled: bool
    dtype: torch.dtype | None
    needs_scaler: bool

    @property
    def name(self) -> str:
        if not self.enabled:
            return "fp32"
        return "bf16" if self.dtype is torch.bfloat16 else "fp16"


def resolve_device(name: str | None) -> torch.device:
    if name in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"device {name!r} was requested but torch.cuda.is_available() is False. "
            "Run scripts/check_gpu.py to see what torch thinks it has."
        )
    return dev


def setup_backends(device: torch.device) -> None:
    if device.type != "cuda":
        return
    # TF32 drops a few mantissa bits on matmul and conv and buys a large speedup
    # from Ampere onwards. Safe here because every reported number is recomputed
    # in fp32 before it leaves the metric accumulator.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    # Shapes are fixed across steps (fixed agent slots, fixed tube size), so
    # cudnn's autotuner pays for itself after the first batch instead of
    # re-benchmarking forever.
    torch.backends.cudnn.benchmark = True


def select_precision(device: torch.device, want_amp: bool) -> Precision:
    if not want_amp or device.type != "cuda":
        return Precision(False, None, False)
    if torch.cuda.is_bf16_supported():
        # bf16 keeps fp32's exponent range, so gradients cannot underflow the
        # way they do in fp16 and no loss scaling is needed.
        return Precision(True, torch.bfloat16, False)
    return Precision(True, torch.float16, True)


def autocast(device: torch.device, precision: Precision):
    if not precision.enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=precision.dtype)


def make_scaler(device: torch.device, precision: Precision) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(device.type, enabled=precision.needs_scaler)


def total_vram_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.get_device_properties(device).total_memory / 1024**3


def free_vram_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.mem_get_info(device)[0] / 1024**3


# Rough ladder, not a measurement: the r3d_18 action branch dominates memory and
# each sample carries max_agents tubes, so this deliberately starts low. Override
# with --batch-size once you have watched a real run's peak VRAM.
_VRAM_LADDER: tuple[tuple[float, int], ...] = (
    (10.0, 4),
    (16.0, 8),
    (24.0, 12),
    (48.0, 24),
    (float("inf"), 48),
)


def auto_batch_size(device: torch.device) -> int:
    if device.type != "cuda":
        return 2
    gb = total_vram_gb(device)
    for limit, bs in _VRAM_LADDER:
        if gb < limit:
            return bs
    return _VRAM_LADDER[-1][1]


def resolve_batch_size(value: int | str, device: torch.device) -> int:
    if isinstance(value, str):
        if value != "auto":
            raise ValueError(f"batch_size must be an int or 'auto', got {value!r}")
        return auto_batch_size(device)
    return int(value)


def auto_num_workers() -> int:
    n = os.cpu_count() or 2
    # Leave the main process and the GPU feed some room. Past ~8 workers the
    # gain flattens out because each one is bound on PNG decode, not on CPU.
    return max(0, min(8, n - 2))


def resolve_num_workers(value: int | str) -> int:
    if isinstance(value, str):
        if value != "auto":
            raise ValueError(f"num_workers must be an int or 'auto', got {value!r}")
        return auto_num_workers()
    return int(value)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024**3


def synchronize(device: torch.device) -> None:
    # Without this every wall-clock and samples/sec number is a lie: the CPU
    # runs ahead of the queued CUDA kernels.
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def describe(device: torch.device, precision: Precision) -> str:
    if device.type != "cuda":
        return f"device: {device}  precision: {precision.name}"
    props = torch.cuda.get_device_properties(device)
    return (
        f"device: {device} ({props.name}, sm_{props.major}{props.minor}, "
        f"{total_vram_gb(device):.1f} GB)  precision: {precision.name}  "
        f"tf32: {torch.backends.cuda.matmul.allow_tf32}"
    )
