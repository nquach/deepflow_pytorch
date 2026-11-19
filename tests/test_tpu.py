"""TPU (torch_xla) support. Most tests run only where torch_xla is installed, e.g. a Colab TPU runtime."""

import importlib.util

import numpy as np
import pytest
import torch

import deepflow
from deepflow.io import resolve_device

HAS_XLA = importlib.util.find_spec("torch_xla") is not None
needs_xla = pytest.mark.skipif(not HAS_XLA, reason="torch_xla is not installed")


def _texture(height=40, width=36, seed=8):
    rng = np.random.RandomState(seed)
    image = rng.randint(0, 256, size=(height, width), dtype=np.uint8)
    shifted = np.roll(np.roll(image, 1, axis=0), -2, axis=1)
    return image, shifted


def _mean_abs(a, b):
    return float(np.mean(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


@pytest.mark.skipif(HAS_XLA, reason="torch_xla is installed")
def test_tpu_without_torch_xla_explains_how_to_install():
    with pytest.raises(ImportError, match="torch_xla"):
        resolve_device("tpu")
    with pytest.raises(ImportError, match="torch_xla"):
        deepflow.DeepFlow(device="xla")


def test_resolve_device_leaves_other_devices_alone():
    assert resolve_device(None) is None
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device(torch.device("meta")) == torch.device("meta")


@pytest.mark.parametrize("device", ["tpu", "xla", "xla:0", torch.device("xla")])
def test_iccv_rejects_tpu(device):
    with pytest.raises(ValueError, match="TPU"):
        deepflow.DeepFlowICCV(device=device)


@needs_xla
def test_tpu_matches_cpu():
    i0, i1 = _texture()
    cpu = deepflow.DeepFlow(device="cpu").calc(i0, i1)
    tpu = deepflow.DeepFlow(device="tpu").calc(i0, i1)
    assert tpu.shape == cpu.shape and tpu.dtype == np.float32
    # TPU sqrt and division round differently from the CPU, as MPS does.
    mean = _mean_abs(cpu, tpu)
    assert mean < 1e-4, f"CPU vs TPU mean abs {mean}"


@needs_xla
def test_tpu_batch_matches_cpu():
    a0, a1 = _texture(seed=4)
    b0, b1 = _texture(seed=5)
    batch0 = torch.from_numpy(np.stack([a0, b0]))[:, None]
    batch1 = torch.from_numpy(np.stack([a1, b1]))[:, None]
    cpu = deepflow.DeepFlow(device="cpu").calc(batch0, batch1)
    tpu = deepflow.DeepFlow(device="tpu").calc(batch0, batch1)
    assert tpu.device.type == "xla"
    mean = _mean_abs(cpu.numpy(), tpu.cpu().numpy())
    assert mean < 1e-4, f"CPU vs TPU mean abs {mean}"


@needs_xla
def test_tpu_compile_matches_lazy():
    settings = dict(device="tpu", fixed_point_iterations=2, sor_iterations=3)
    i0, i1 = _texture()
    lazy = deepflow.DeepFlow(**settings).calc(i0, i1)
    compiled = deepflow.DeepFlow(compile=True, **settings).calc(i0, i1)
    mean = _mean_abs(lazy, compiled)
    assert mean < 1e-5, f"lazy vs compiled mean abs {mean}"


@needs_xla
def test_tensors_already_on_tpu_stay_there():
    device = resolve_device("tpu")
    i0, i1 = (torch.from_numpy(image).to(device) for image in _texture())
    flow = deepflow.DeepFlow(fixed_point_iterations=1, sor_iterations=2).calc(i0, i1)
    assert flow.device.type == "xla"
    assert flow.shape == (2, *i0.shape)


@needs_xla
def test_iccv_rejects_tensors_on_tpu():
    device = resolve_device("tpu")
    i0, i1 = (torch.from_numpy(image).to(device) for image in _texture())
    with pytest.raises(ValueError, match="TPU"):
        deepflow.DeepFlowICCV().calc(i0, i1)
