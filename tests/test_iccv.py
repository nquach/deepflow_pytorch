"""DeepMatching pieces and the ICCV 2013 entry point."""

import numpy as np
import pytest
import torch

import deepflow
from deepflow.matching import aggregate_pooled, atomic_correlation, deep_match
from deepflow.variational import variational_refine


def test_translated_patch_peaks_at_the_shift():
    reference = torch.zeros(20, 20)
    pattern = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    reference[4:8, 4:8] = pattern
    shift_y, shift_x = 1, 3
    target = torch.zeros(20, 20)
    target[4 + shift_y : 8 + shift_y, 4 + shift_x : 8 + shift_x] = pattern
    correlation = atomic_correlation(
        reference,
        target,
        descriptor_smooth=0.0,
        sigmoid_slope=0.2,
        descriptor_regularizer=0.3,
    )
    # Centers are 2, 6, 10, 14. The pattern sits on the patch centered at 6.
    peak = int(correlation[1, 1].argmax())
    width = correlation.shape[-1]
    peak_y, peak_x = divmod(peak, width)
    assert (peak_y, peak_x) == (6 + shift_y, 6 + shift_x)


def test_aggregation_renormalizes_a_missing_border_child():
    pooled = torch.zeros(2, 2, 6, 6)
    pooled[0, 0] = 1
    pooled[0, 1] = 2
    pooled[1, 0] = 3
    pooled[1, 1] = 4
    power = 1.4
    merged = aggregate_pooled(pooled, power)
    assert merged.shape == (1, 1, 6, 6)
    assert merged[0, 0, 3, 3].item() == pytest.approx(2.5**power)
    assert merged[0, 0, 0, 3].item() == pytest.approx(1.5**power)
    assert merged[0, 0, 0, 0].item() == pytest.approx(1.0**power)
    assert merged[0, 0, 5, 5].item() == pytest.approx(4.0**power)


def test_match_term_pulls_toward_the_correspondence():
    image = torch.ones(1, 1, 24, 24)
    flow_u = torch.zeros_like(image)
    flow_v = torch.zeros_like(image)
    match_u = torch.full_like(image, 4.0)
    match_v = torch.zeros_like(image)
    weight = torch.ones_like(image)
    out_u, out_v = variational_refine(
        image,
        image,
        flow_u,
        flow_v,
        alpha=0.01,
        delta=0.0,
        gamma=0.0,
        fixed_point_iterations=5,
        sor_iterations=25,
        match_u=match_u,
        match_v=match_v,
        match_weight=weight,
        beta=40.0,
    )
    center = out_u[0, 0, 6:18, 6:18]
    assert float(center.mean()) > 2.5
    assert float(out_v.abs().mean()) < 0.5


def _square(shift=16, size=48):
    reference = np.full((size, size), 30, np.float32)
    target = np.full((size, size), 30, np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    y0, x0, side = 14, 8, 12
    source = (yy >= y0) & (yy < y0 + side) & (xx >= x0) & (xx < x0 + side)
    destination = (yy >= y0) & (yy < y0 + side) & (xx >= x0 + shift) & (xx < x0 + side + shift)
    reference[source] = 50 + 8 * np.sin((yy[source] - y0) * 0.9) + 12 * np.cos((xx[source] - x0) * 0.7)
    target[destination] = (
        50
        + 8 * np.sin((yy[destination] - y0) * 0.9)
        + 12 * np.cos((xx[destination] - (x0 + shift)) * 0.7)
    )
    return reference, target, source


def _endpoint(flow, shift, region):
    error = np.sqrt((flow[..., 0] - shift) ** 2 + flow[..., 1] ** 2)
    return float(error[region].mean())


def test_large_shift_beats_opencv_deepflow():
    shift = 16
    reference, target, region = _square(shift=shift)
    # One pyramid level. A 16px shift of a 12px square does not overlap, so the
    # coarse variational linearization cannot see it.
    common = dict(min_size=46)
    plain = deepflow.DeepFlow(**common).calc(reference, target)
    paper = deepflow.DeepFlowICCV(**common, match_max_side=None).calc(reference, target)
    plain_error = _endpoint(plain, shift, region)
    paper_error = _endpoint(paper, shift, region)
    assert paper_error < plain_error, f"ICCV {paper_error}, OpenCV-compatible {plain_error}"


def test_deep_match_recovers_the_square_shift():
    shift = 16
    reference, target, region = _square(shift=shift)
    field = deep_match(
        torch.from_numpy(reference)[None, None],
        torch.from_numpy(target)[None, None],
        match_max_side=None,
    )
    mask = field.mask[0, 0].numpy() > 0
    covered = region & mask
    assert int(covered.sum()) > 10
    mean_u = float(field.u[0, 0].numpy()[covered].mean())
    mean_v = float(field.v[0, 0].numpy()[covered].mean())
    assert abs(mean_u - shift) < 3.0, f"mean u {mean_u}"
    assert abs(mean_v) < 3.0, f"mean v {mean_v}"


_FAST = dict(min_size=28, fixed_point_iterations=2, sor_iterations=2, max_layers=2, match_max_side=None)


def _texture(height, width, seed):
    rng = np.random.RandomState(seed)
    image = rng.randint(0, 256, size=(height, width), dtype=np.uint8)
    shifted = np.roll(np.roll(image, 1, axis=0), -2, axis=1)
    return image, shifted


def test_iccv_uint8_matches_float_range():
    i0, i1 = _texture(32, 32, seed=3)
    as_float = (i0.astype(np.float32), i1.astype(np.float32))
    integer = deepflow.DeepFlowICCV(**_FAST).calc(i0, i1)
    floating = deepflow.DeepFlowICCV(**_FAST).calc(*as_float)
    mean = float(np.mean(np.abs(integer.astype(np.float64) - floating.astype(np.float64))))
    assert mean < 1e-5, f"mean abs {mean}"


def test_iccv_layouts_and_ignored_flow():
    i0, i1 = _texture(32, 30, seed=6)
    numpy_flow = deepflow.DeepFlowICCV(**_FAST).calc(i0, i1)
    channel = deepflow.DeepFlowICCV(**_FAST).calc(i0[:, :, None], i1[:, :, None])
    assert numpy_flow.shape == (32, 30, 2)
    assert numpy_flow.dtype == np.float32
    assert np.allclose(numpy_flow, channel)

    ignored = np.ones_like(numpy_flow)
    again = deepflow.DeepFlowICCV(**_FAST).calc(i0, i1, flow=ignored)
    assert np.allclose(numpy_flow, again)

    t0 = torch.from_numpy(i0)
    t1 = torch.from_numpy(i1)
    hw = deepflow.DeepFlowICCV(**_FAST).calc(t0, t1)
    chw = deepflow.DeepFlowICCV(**_FAST).calc(t0[None], t1[None])
    assert hw.shape == (2, 32, 30)
    assert torch.allclose(hw, chw)

    batch0 = torch.stack([t0, torch.from_numpy(_texture(32, 30, seed=4)[0])])[:, None]
    batch1 = torch.stack([t1, torch.from_numpy(_texture(32, 30, seed=4)[1])])[:, None]
    batched = deepflow.DeepFlowICCV(**_FAST).calc(batch0, batch1)
    assert batched.shape == (2, 2, 32, 30)


def test_iccv_color_input_is_rejected():
    color = np.zeros((16, 16, 3), np.uint8)
    with pytest.raises(ValueError, match="single-channel"):
        deepflow.DeepFlowICCV().calc(color, color)


def test_iccv_factory_matches_class():
    i0, i1 = _texture(32, 30, seed=7)
    a = deepflow.create_opt_flow_deep_flow_iccv(**_FAST).calc(i0, i1)
    b = deepflow.DeepFlowICCV(**_FAST).calc(i0, i1)
    assert np.allclose(a, b)
