"""CPU, MPS, and CUDA must agree on the same pair."""

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

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


class _HostTensorWatch(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        flat, _ = tree_flatten((args, kwargs, out))
        if any(isinstance(t, torch.Tensor) and t.device.type == "cpu" for t in flat):
            self.ops.add(str(func))
        return out


def test_estimate_touches_no_host_tensor_after_warmup():
    # CUDA graph capture forbids host-to-device copies, so after the first call
    # for a shape the pipeline must stay entirely on the compute device.
    model = deepflow.DeepFlow(fixed_point_iterations=1, sor_iterations=1)
    i0 = torch.empty(2, 1, 60, 70, device="meta")
    i1 = torch.empty(2, 1, 60, 70, device="meta")
    with torch.no_grad():
        model._estimate(i0, i1)
        watch = _HostTensorWatch()
        with watch:
            model._estimate(i0, i1)
    assert not watch.ops, sorted(watch.ops)


def test_cuda_graph_option_is_ignored_on_cpu():
    i0, i1 = _texture()
    plain = deepflow.DeepFlow(device="cpu").calc(i0, i1)
    graphed = deepflow.DeepFlow(device="cpu", cuda_graph=True).calc(i0, i1)
    assert np.array_equal(plain, graphed)


def test_cuda_graph_matches_eager_and_refreshes_inputs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    graphed = deepflow.DeepFlow(device="cuda", cuda_graph=True)
    eager = deepflow.DeepFlow(device="cuda", cuda_graph=False)
    first = _texture(40, 36)
    rng = np.random.RandomState(9)
    other = rng.randint(0, 256, size=(40, 36), dtype=np.uint8)
    second = (other, np.roll(other, 2, axis=1))
    for pair in (first, second):
        actual = graphed.calc(*pair)
        expected = eager.calc(*pair)
        assert np.abs(actual - expected).max() < 1e-6
    assert len(graphed._graphs) == 1
    assert not graphed._graph_failed


def test_cuda_graph_captures_each_shape():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    model = deepflow.DeepFlow(device="cuda")
    eager = deepflow.DeepFlow(device="cuda", cuda_graph=False)
    for shape in ((32, 32), (40, 36), (32, 32)):
        pair = _texture(*shape)
        assert np.abs(model.calc(*pair) - eager.calc(*pair)).max() < 1e-6
    assert len(model._graphs) == 2
    batch0 = torch.rand(3, 1, 32, 30, device="cuda") * 255
    batch1 = torch.roll(batch0, 1, dims=-1)
    assert torch.allclose(model.calc(batch0, batch1), eager.calc(batch0, batch1), atol=1e-6)
    assert len(model._graphs) == 3
