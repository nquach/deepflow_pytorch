"""CPU, MPS, and CUDA must agree on the same pair."""

import numpy as np
import pytest
import torch

import deepflow


def _texture(height=32, width=32):
    rng = np.random.RandomState(8)
    image = rng.randint(0, 256, size=(height, width), dtype=np.uint8)
    shifted = np.roll(np.roll(image, 1, axis=0), -2, axis=1)
    return image, shifted


def _mean_abs(a, b):
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def test_cpu_matches_mps():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    i0, i1 = _texture()
    cpu = deepflow.DeepFlow(device="cpu").calc(i0, i1)
    mps = deepflow.DeepFlow(device="mps").calc(i0, i1)
    mean = _mean_abs(cpu, mps)
    assert mean < 1e-4, f"CPU vs MPS mean abs {mean}"


def _smooth_pair(height=28, width=28):
    # A smooth shift keeps one correspondence, so the device comparison is the
    # solver rather than a coin-flip between equal match scores.
    yy, xx = np.mgrid[0:height, 0:width]
    image = (100 + 40 * np.sin(xx / 3.0) + 25 * np.cos(yy / 2.5)).astype(np.float32)
    target = (100 + 40 * np.sin((xx - 1.5) / 3.0) + 25 * np.cos((yy - 0.5) / 2.5)).astype(np.float32)
    return image, target


def test_iccv_cpu_matches_mps():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    i0, i1 = _smooth_pair()
    cpu = deepflow.DeepFlowICCV(device="cpu").calc(i0, i1)
    mps = deepflow.DeepFlowICCV(device="mps").calc(i0, i1)
    mean = _mean_abs(cpu, mps)
    assert mean < 1e-4, f"CPU vs MPS mean abs {mean}"


def test_iccv_cpu_matches_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    i0, i1 = _smooth_pair()
    cpu = deepflow.DeepFlowICCV(device="cpu").calc(i0, i1)
    cuda = deepflow.DeepFlowICCV(device="cuda").calc(i0, i1)
    mean = _mean_abs(cpu, cuda)
    assert mean < 1e-4, f"CPU vs CUDA mean abs {mean}"


def test_cpu_matches_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    i0, i1 = _texture()
    cpu = deepflow.DeepFlow(device="cpu").calc(i0, i1)
    cuda = deepflow.DeepFlow(device="cuda").calc(i0, i1)
    mean = _mean_abs(cpu, cuda)
    assert mean < 1e-4, f"CPU vs CUDA mean abs {mean}"
