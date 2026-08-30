#!/usr/bin/env python
"""Run this first. It tells you whether the machine can actually train this.

Prints what torch sees, what precision the training code will pick on this
hardware, and runs a small matmul so you can tell a working GPU from one that
merely imports.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from titan.device import (  # noqa: E402
    auto_batch_size,
    auto_num_workers,
    resolve_device,
    select_precision,
    setup_backends,
)


def benchmark(device: torch.device, n: int = 4096, iters: int = 20) -> float:
    """TFLOP/s on a square fp32 matmul. Rough, but catches a dead GPU."""
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    for _ in range(3):
        a @ b
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    return (2 * n**3 * iters) / elapsed / 1e12


def main() -> int:
    print("== torch ==")
    print(f"  torch            {torch.__version__}")
    print(f"  built for CUDA   {torch.version.cuda or 'cpu-only build'}")
    print(f"  cuda.is_available {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\nNo CUDA device visible. Training will run on CPU and will be very slow.")
        print("Check: nvidia-smi works, and the installed torch is a CUDA build")
        print("(a cpu-only wheel reports 'cpu-only build' above).")
        device = torch.device("cpu")
        setup_backends(device)
        print(f"\n== benchmark ==\n  fp32 matmul      {benchmark(device, n=2048, iters=5):.2f} TFLOP/s")
        return 1

    device = resolve_device("auto")
    props = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)

    print("\n== device ==")
    print(f"  name             {props.name}")
    print(f"  compute cap      {props.major}.{props.minor}")
    print(f"  multiprocessors  {props.multi_processor_count}")
    print(f"  total VRAM       {total / 1024**3:.2f} GB")
    print(f"  free VRAM        {free / 1024**3:.2f} GB")
    print(f"  bf16 supported   {torch.cuda.is_bf16_supported()}")

    setup_backends(device)
    precision = select_precision(device, want_amp=True)

    print("\n== precision settings this repo will use ==")
    print(f"  autocast dtype   {precision.name}")
    print(f"  grad scaler      {'on (fp16 needs it)' if precision.needs_scaler else 'off'}")
    print(f"  matmul tf32      {torch.backends.cuda.matmul.allow_tf32}")
    print(f"  cudnn tf32       {torch.backends.cudnn.allow_tf32}")
    print(f"  cudnn benchmark  {torch.backends.cudnn.benchmark}")
    print(f"  matmul precision {torch.get_float32_matmul_precision()}")

    print("\n== defaults picked for this machine ==")
    print(f"  batch size       {auto_batch_size(device)}   (override with --batch-size)")
    print(f"  dataloader workers {auto_num_workers()}   (override with --num-workers)")

    print("\n== benchmark ==")
    print(f"  fp32 4096^2 matmul {benchmark(device):.2f} TFLOP/s")
    print(f"  peak VRAM in bench {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")

    print("\nLooks usable. Next: pytest -q tests, then the synthetic smoke test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
