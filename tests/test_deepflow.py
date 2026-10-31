"""Full pyramid against cv2.optflow.createOptFlow_DeepFlow."""

import numpy as np
import pytest
import torch

cv2 = pytest.importorskip("cv2")
pytest.importorskip("cv2.optflow", reason="opencv-contrib is required for DeepFlow")

import deepflow
from deepflow.deepflow import build_pyramid


def _texture(height, width, seed=1):
    rng = np.random.RandomState(seed)
    image = rng.randint(0, 256, size=(height, width), dtype=np.uint8)
    shifted = np.roll(np.roll(image, 1, axis=0), 2, axis=1)
    return image, shifted


def _errors(actual, expected):
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return float(diff.mean()), float(diff.max())


def _opencv_flow(i0, i1):
    solver = cv2.optflow.createOptFlow_DeepFlow()
    return solver.calc(i0, i1, None)


def test_pyramid_sizes_match_opencv_rounding():
    image = torch.arange(48 * 41, dtype=torch.float32).reshape(1, 1, 48, 41)
    levels = build_pyramid(image, 0.95, 25, 200)
    prev = image
    expected_shapes = [tuple(image.shape[-2:])]
    for _ in range(200):
        h, w = prev.shape[-2:]
        nh = int(np.floor(h * 0.95 + 0.5))
        nw = int(np.floor(w * 0.95 + 0.5))
        if nh <= 25 or nw <= 25:
            break
        prev = torch.empty(1, 1, nh, nw)
        expected_shapes.append((nh, nw))
    assert [tuple(level.shape[-2:]) for level in levels] == expected_shapes


def test_small_image_matches_opencv():
    i0, i1 = _texture(40, 36)
    expected = _opencv_flow(i0, i1)
    actual = deepflow.DeepFlow().calc(i0, i1)
    mean, maximum = _errors(actual, expected)
    assert actual.shape == expected.shape
    assert actual.dtype == np.float32
    assert mean < 1e-5 and maximum < 1e-4, f"mean abs {mean}, max abs {maximum}"


def test_larger_image_matches_opencv():
    i0, i1 = _texture(64, 70, seed=2)
    expected = _opencv_flow(i0, i1)
    actual = deepflow.DeepFlow().calc(i0, i1, flow=np.ones_like(expected))
    mean, maximum = _errors(actual, expected)
    assert mean < 1e-5 and maximum < 1e-4, f"mean abs {mean}, max abs {maximum}"


def test_numpy_channel_axis_matches_2d():
    i0, i1 = _texture(32, 30, seed=6)
    flat = deepflow.DeepFlow().calc(i0, i1)
    channel = deepflow.DeepFlow().calc(i0[:, :, None], i1[:, :, None])
    assert np.array_equal(flat, channel)


def test_uint8_matches_float_range():
    i0, i1 = _texture(32, 32, seed=3)
    as_float = (i0.astype(np.float32), i1.astype(np.float32))
    a = deepflow.DeepFlow().calc(i0, i1)
    b = deepflow.DeepFlow().calc(*as_float)
    mean, maximum = _errors(a, b)
    assert mean < 1e-6, f"mean abs {mean}, max abs {maximum}"


def test_torch_batch_matches_separate_calls():
    a0, a1 = _texture(32, 30, seed=4)
    b0, b1 = _texture(32, 30, seed=5)
    batch0 = torch.from_numpy(np.stack([a0, b0]))[:, None]
    batch1 = torch.from_numpy(np.stack([a1, b1]))[:, None]
    batched = deepflow.DeepFlow().calc(batch0, batch1)
    separate = np.stack([
        deepflow.DeepFlow().calc(a0, a1).transpose(2, 0, 1),
        deepflow.DeepFlow().calc(b0, b1).transpose(2, 0, 1),
    ])
    mean, maximum = _errors(batched.numpy(), separate)
    assert batched.shape == (2, 2, 32, 30)
    assert mean < 1e-6, f"mean abs {mean}, max abs {maximum}"


def test_single_channel_torch_layouts():
    i0, i1 = _texture(32, 30, seed=6)
    t0 = torch.from_numpy(i0)
    t1 = torch.from_numpy(i1)
    hw = deepflow.DeepFlow().calc(t0, t1)
    chw = deepflow.DeepFlow().calc(t0[None], t1[None])
    assert hw.shape == (2, 32, 30)
    assert torch.allclose(hw, chw)


def test_color_input_is_rejected():
    color = np.zeros((16, 16, 3), np.uint8)
    with pytest.raises(ValueError, match="single-channel"):
        deepflow.DeepFlow().calc(color, color)
    tensor = torch.zeros(2, 3, 16, 16)
    with pytest.raises(ValueError, match="single-channel"):
        deepflow.DeepFlow().calc(tensor, tensor)


def test_factory_matches_class():
    i0, i1 = _texture(32, 30, seed=7)
    a = deepflow.create_opt_flow_deep_flow().calc(i0, i1)
    b = deepflow.DeepFlow().calc(i0, i1)
    assert np.array_equal(a, b)
