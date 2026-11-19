"""Image ops matched to the OpenCV calls used by DeepFlow."""

from __future__ import annotations

import functools
import math

import torch
import torch.nn.functional as F


def gaussian_kernel_size(sigma: float) -> int:
    """Odd window used by ``cv::GaussianBlur`` when ``ksize = floor(3*sigma)*2+1``."""
    return int(math.floor(3.0 * float(sigma))) * 2 + 1


def gaussian_kernel(ksize: int, sigma: float, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Match ``cv::getGaussianKernel`` for an explicit ``sigma > 0`` and ``CV_32F``.

    OpenCV evaluates the Gaussian in double, stores each weight as float32, then
    normalizes those float32 values in double precision and casts back.
    """
    if ksize < 1 or ksize % 2 == 0:
        raise ValueError(f"Gaussian kernel size must be a positive odd integer, got {ksize}")
    sigma_x = float(sigma) if sigma > 0 else ((ksize - 1) * 0.5 - 1.0) * 0.3 + 0.8
    scale2 = -0.5 / (sigma_x * sigma_x)
    centers = torch.arange(ksize, dtype=torch.float64) - (ksize - 1) * 0.5
    weights = torch.exp(scale2 * centers * centers).to(torch.float32)
    weights = weights * (1.0 / weights.sum(dtype=torch.float64))
    return weights.to(dtype=torch.float32).to(device=device, dtype=dtype)


def _reflect101(index: torch.Tensor, length: int) -> torch.Tensor:
    """OpenCV ``BORDER_REFLECT_101`` (``gfedcb|abcdefgh|gfedcba``)."""
    if length <= 1:
        return torch.zeros_like(index)
    period = 2 * length - 2
    mod = torch.remainder(index, period)
    return torch.where(mod < length, mod, period - mod)


def _replicate(index: torch.Tensor, length: int) -> torch.Tensor:
    return index.clamp(0, length - 1)


@functools.lru_cache(maxsize=64)
def _gaussian_taps(ksize: int, sigma: float) -> tuple[float, ...]:
    """float32 Gaussian weights as Python floats, so filtering needs no device copy."""
    return tuple(float(w) for w in gaussian_kernel(ksize, sigma, dtype=torch.float32, device=torch.device("cpu")))


def _pad_axis(image: torch.Tensor, radius: int, *, axis: int, border: str) -> torch.Tensor:
    pad = (radius, radius, 0, 0) if axis == 3 else (0, 0, radius, radius)
    if border == "reflect101":
        return F.pad(image, pad, mode="reflect")
    if border == "replicate":
        return F.pad(image, pad, mode="replicate")
    raise ValueError(f"unknown border {border}")


def _correlate_taps(image: torch.Tensor, taps: tuple[float, ...], *, axis: int, border: str) -> torch.Tensor:
    """Separable correlation with scalar taps, accumulated in tap order."""
    if axis not in (2, 3):
        raise ValueError("axis must be 2 (vertical) or 3 (horizontal)")
    length = image.shape[axis]
    radius = len(taps) // 2
    if border == "reflect101" and length <= radius:
        # F.pad's reflect needs the pad to be shorter than the axis.
        kernel = torch.tensor(taps, dtype=image.dtype, device=image.device)
        return _correlate_axis(image, kernel, axis=axis, border=border)
    padded = _pad_axis(image, radius, axis=axis, border=border)
    total = padded.narrow(axis, 0, length) * taps[0]
    for index in range(1, len(taps)):
        total = total + padded.narrow(axis, index, length) * taps[index]
    return total


def _correlate_axis(image: torch.Tensor, kernel: torch.Tensor, *, axis: int, border: str) -> torch.Tensor:
    """Separable correlation along height (axis=2) or width (axis=3), by gathering."""
    if axis not in (2, 3):
        raise ValueError("axis must be 2 (vertical) or 3 (horizontal)")
    length = image.shape[axis]
    radius = kernel.numel() // 2
    offsets = torch.arange(kernel.numel(), device=image.device) - radius
    coords = torch.arange(length, device=image.device)
    sample = coords.unsqueeze(0) + offsets.unsqueeze(1)
    if border == "reflect101":
        sample = _reflect101(sample, length)
    elif border == "replicate":
        sample = _replicate(sample, length)
    else:
        raise ValueError(f"unknown border {border}")

    if axis == 3:
        gathered = image[:, :, :, sample]
        weight = kernel.view(1, 1, 1, -1, 1)
        return (gathered * weight).sum(dim=3)
    gathered = image[:, :, sample, :]
    weight = kernel.view(1, 1, -1, 1, 1)
    return (gathered * weight).sum(dim=2)


def gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """``cv::GaussianBlur`` with ``ksize = floor(3*sigma)*2+1`` and ``BORDER_REFLECT_101``.

    ``image`` is ``N,C,H,W`` float. A kernel of size 1 leaves the image unchanged.
    """
    ksize = gaussian_kernel_size(sigma)
    if ksize <= 1:
        return image
    taps = _gaussian_taps(ksize, float(sigma))
    blurred = _correlate_taps(image, taps, axis=3, border="reflect101")
    return _correlate_taps(blurred, taps, axis=2, border="reflect101")


def _central_difference(image: torch.Tensor, *, axis: int) -> torch.Tensor:
    """OpenCV ``Sobel(..., ksize=1, scale=1, BORDER_REPLICATE)``.

    The ksize-1 kernel is the unnormalized central difference ``[-1, 0, 1]``.
    """
    length = image.shape[axis]
    padded = _pad_axis(image, 1, axis=axis, border="replicate")
    return padded.narrow(axis, 2, length) - padded.narrow(axis, 0, length)


def sobel_x(image: torch.Tensor) -> torch.Tensor:
    return _central_difference(image, axis=3)


def sobel_y(image: torch.Tensor) -> torch.Tensor:
    return _central_difference(image, axis=2)


def _bilinear_sample(image: torch.Tensor, map_x: torch.Tensor, map_y: torch.Tensor) -> torch.Tensor:
    """Bilinear sample with ``BORDER_REPLICATE``.

    ``map_x`` and ``map_y`` are float pixel coordinates shaped ``N,H,W`` (or
    broadcastable to the output spatial size). ``image`` is ``N,C,H,W``.
    """
    n, channels, height, width = image.shape
    map_x = map_x.to(dtype=torch.float32)
    map_y = map_y.to(dtype=torch.float32)
    map_x = torch.nan_to_num(map_x, nan=0.0, posinf=float(width - 1), neginf=0.0)
    map_y = torch.nan_to_num(map_y, nan=0.0, posinf=float(height - 1), neginf=0.0)
    # Coordinates outside the image collapse to the border pixel under replicate.
    map_x = map_x.clamp(-1.0, float(width - 1))
    map_y = map_y.clamp(-1.0, float(height - 1))

    x0 = torch.floor(map_x)
    y0 = torch.floor(map_y)
    ax = (map_x - x0).to(dtype=image.dtype)
    ay = (map_y - y0).to(dtype=image.dtype)
    x0i = x0.clamp(0, width - 1).to(torch.long)
    y0i = y0.clamp(0, height - 1).to(torch.long)
    x1i = (x0 + 1).clamp(0, width - 1).to(torch.long)
    y1i = (y0 + 1).clamp(0, height - 1).to(torch.long)

    flat = image.reshape(n, channels, height * width)

    def gather(yy: torch.Tensor, xx: torch.Tensor) -> torch.Tensor:
        spatial = yy.shape[-2] * yy.shape[-1]
        index = (yy * width + xx).reshape(n, 1, spatial).expand(n, channels, spatial)
        return flat.gather(2, index).reshape(n, channels, yy.shape[-2], yy.shape[-1])

    v00 = gather(y0i, x0i)
    v01 = gather(y0i, x1i)
    v10 = gather(y1i, x0i)
    v11 = gather(y1i, x1i)
    ax = ax[:, None]
    ay = ay[:, None]
    return (
        v00 * (1 - ax) * (1 - ay)
        + v01 * ax * (1 - ay)
        + v10 * (1 - ax) * ay
        + v11 * ax * ay
    )


def resize_bilinear(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """``cv::resize(..., INTER_LINEAR)`` using the OpenCV 4 pixel-center mapping.

    ``size`` is ``(height, width)``. Source coordinate of output pixel ``x`` is
    ``(x + 0.5) * (src/dst) - 0.5``, with samples clamped like ``BORDER_REPLICATE``.
    """
    out_h, out_w = int(size[0]), int(size[1])
    if out_h < 1 or out_w < 1:
        raise ValueError(f"output size must be positive, got {(out_h, out_w)}")
    _, _, in_h, in_w = image.shape
    if out_h == in_h and out_w == in_w:
        return image

    map_x, map_y = _resize_coords(in_h, in_w, out_h, out_w, image.device)
    map_x = map_x.view(1, 1, out_w).expand(image.shape[0], out_h, out_w)
    map_y = map_y.view(1, out_h, 1).expand(image.shape[0], out_h, out_w)
    return _bilinear_sample(image, map_x, map_y)


@functools.lru_cache(maxsize=1024)
def _resize_coords(
    in_h: int, in_w: int, out_h: int, out_w: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Source coordinates for ``resize_bilinear``, kept on ``device``.

    Cached so repeated calls (and CUDA graph capture) do no host-to-device copy.
    Coordinates are built in float64 on CPU. MPS has no float64, and OpenCV
    uses double for the scale before the sample coordinates are stored as float.
    """
    inv_x = float(in_w) / float(out_w)
    inv_y = float(in_h) / float(out_h)
    xs = (torch.arange(out_w, dtype=torch.float64) + 0.5) * inv_x - 0.5
    ys = (torch.arange(out_h, dtype=torch.float64) + 0.5) * inv_y - 0.5
    return xs.to(dtype=torch.float32, device=device), ys.to(dtype=torch.float32, device=device)


def remap_bilinear(image: torch.Tensor, map_x: torch.Tensor, map_y: torch.Tensor) -> torch.Tensor:
    """``cv::remap(..., INTER_LINEAR, BORDER_REPLICATE)`` in floating point.

    ``map_x`` / ``map_y`` use OpenCV's pixel coordinate system (integer coordinates
    are pixel centers) and have shape ``N,H,W`` or ``N,1,H,W``.
    """
    if map_x.dim() == 4:
        map_x = map_x.squeeze(1)
    if map_y.dim() == 4:
        map_y = map_y.squeeze(1)
    return _bilinear_sample(image, map_x, map_y)
