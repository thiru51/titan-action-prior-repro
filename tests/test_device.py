import pytest
import torch

from titan.device import (
    Precision,
    auto_batch_size,
    autocast,
    make_scaler,
    resolve_batch_size,
    resolve_device,
    resolve_num_workers,
    select_precision,
)


def test_auto_device_matches_availability():
    dev = resolve_device("auto")
    assert dev.type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_explicit_cpu_is_honoured_even_with_a_gpu_present():
    assert resolve_device("cpu").type == "cpu"


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a machine without CUDA")
def test_requesting_cuda_without_cuda_fails_loudly():
    with pytest.raises(RuntimeError):
        resolve_device("cuda")


def test_amp_is_off_on_cpu():
    # autocast on CPU would silently pick a different dtype policy; the training
    # step is only ever meant to run mixed precision on CUDA.
    p = select_precision(torch.device("cpu"), want_amp=True)
    assert p.enabled is False
    assert p.name == "fp32"


def test_no_amp_flag_gives_fp32():
    p = select_precision(resolve_device("auto"), want_amp=False)
    assert p.enabled is False


def test_scaler_only_engages_for_fp16():
    # bf16 has fp32's exponent range, so loss scaling is not just unnecessary,
    # it is a source of confusion if it silently runs.
    dev = torch.device("cpu")
    assert make_scaler(dev, Precision(True, torch.float16, True)).is_enabled()
    assert not make_scaler(dev, Precision(True, torch.bfloat16, False)).is_enabled()
    assert not make_scaler(dev, Precision(False, None, False)).is_enabled()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_precision_picks_bf16_when_supported():
    p = select_precision(resolve_device("auto"), want_amp=True)
    assert p.enabled is True
    if torch.cuda.is_bf16_supported():
        assert p.dtype is torch.bfloat16 and p.needs_scaler is False
    else:
        assert p.dtype is torch.float16 and p.needs_scaler is True


def test_disabled_precision_autocast_is_a_no_op():
    dev = torch.device("cpu")
    with autocast(dev, Precision(False, None, False)):
        out = torch.ones(2, 2) @ torch.ones(2, 2)
    assert out.dtype is torch.float32


def test_explicit_sizes_win_over_auto():
    dev = resolve_device("auto")
    assert resolve_batch_size(37, dev) == 37
    assert resolve_num_workers(3) == 3


def test_auto_sizes_are_sane():
    dev = resolve_device("auto")
    assert resolve_batch_size("auto", dev) == auto_batch_size(dev)
    assert auto_batch_size(dev) >= 1
    assert resolve_num_workers("auto") >= 0


def test_bad_auto_string_rejected():
    with pytest.raises(ValueError):
        resolve_batch_size("as many as fit", resolve_device("auto"))
    with pytest.raises(ValueError):
        resolve_num_workers("lots")
