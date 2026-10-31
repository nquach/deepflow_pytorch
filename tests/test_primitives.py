"""Compare Gaussian, Sobel, resize, and remap against OpenCV."""

import numpy as np
import pytest
import torch

cv2 = pytest.importorskip("cv2")

from deepflow.ops import gaussian_blur, remap_bilinear, resize_bilinear, sobel_x, sobel_y


def _to_nchw(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image)).view(1, 1, *image.shape)


def _max_abs(actual: torch.Tensor, expected: np.ndarray) -> float:
    got = actual.detach().cpu().numpy().reshape(expected.shape)
    return float(np.max(np.abs(got.astype(np.float64) - expected.astype(np.float64))))


def _pattern(height: int, width: int) -> np.ndarray:
    ys, xs = np.mgrid[0:height, 0:width]
    image = (np.sin(xs * 0.37) * 40 + np.cos(ys * 0.21) * 30 + xs * 0.5 + ys * 0.25).astype(np.float32)
    return image


def test_gaussian_matches_opencv():
    image = _pattern(40, 37)
    sigma = 0.6
    ksize = int(np.floor(3 * sigma)) * 2 + 1
    expected = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    actual = gaussian_blur(_to_nchw(image), sigma)
    # A couple of float32 ulps versus OpenCV's separable filter accumulation.
    assert _max_abs(actual, expected) < 2e-5


def test_sobel_matches_opencv():
    image = _pattern(33, 31)
    expected_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1, scale=1, delta=0, borderType=cv2.BORDER_REPLICATE)
    expected_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1, scale=1, delta=0, borderType=cv2.BORDER_REPLICATE)
    tensor = _to_nchw(image)
    assert _max_abs(sobel_x(tensor), expected_x) < 1e-5
    assert _max_abs(sobel_y(tensor), expected_y) < 1e-5


def test_resize_matches_opencv_odd_and_even():
    image = _pattern(48, 41)
    for size in ((25, 22), (63, 70), (48, 41)):
        expected = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
        actual = resize_bilinear(_to_nchw(image), size)
        err = _max_abs(actual, expected)
        assert err < 2e-5, f"size {size} max abs {err}"


def test_remap_matches_opencv():
    image = _pattern(28, 26)
    ys, xs = np.mgrid[0:28, 0:26].astype(np.float32)
    map_x = xs + 1.35
    map_y = ys - 0.8
    map_x[0, 0] = -2.2
    map_y[-1, -1] = 40.0
    expected = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    actual = remap_bilinear(
        _to_nchw(image),
        torch.from_numpy(map_x)[None],
        torch.from_numpy(map_y)[None],
    )
    err = _max_abs(actual, expected)
    assert err < 2e-5, f"remap max abs {err}"
