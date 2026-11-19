"""NumPy and torch layouts shared by both DeepFlow entry points."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def is_numpy(value: Any) -> bool:
    return isinstance(value, np.ndarray)


def color_error(shape: tuple[int, ...]) -> ValueError:
    return ValueError(
        "DeepFlow expects a single-channel grayscale image, "
        f"got shape {shape}. Convert color images the same way you would before "
        "calling cv2.optflow.createOptFlow_DeepFlow (it rejects anything but 1 channel)."
    )


def numpy_to_nchw(image: np.ndarray) -> tuple[torch.Tensor, str]:
    if image.ndim == 2:
        tensor = torch.from_numpy(np.ascontiguousarray(image))[None, None]
        return tensor, "hw"
    if image.ndim == 3 and image.shape[2] == 1:
        tensor = torch.from_numpy(np.ascontiguousarray(image[:, :, 0]))[None, None]
        return tensor, "hw"
    if image.ndim == 3 and image.shape[2] != 1:
        raise color_error(tuple(image.shape))
    raise ValueError(
        "NumPy images must have shape (H, W) or (H, W, 1), "
        f"got {tuple(image.shape)}"
    )


def torch_to_nchw(image: torch.Tensor) -> tuple[torch.Tensor, str]:
    if image.ndim == 2:
        # (H, W) is one grayscale image, returned as (2, H, W), same as (1, H, W).
        return image[None, None], "chw"
    if image.ndim == 3:
        if image.shape[0] == 1:
            return image[None], "chw"
        return image[:, None], "nhw"
    if image.ndim == 4:
        if image.shape[1] != 1:
            raise color_error(tuple(image.shape))
        return image, "nchw"
    raise ValueError(
        "Torch images must have shape (H, W), (1, H, W), (N, H, W), or (N, 1, H, W), "
        f"got {tuple(image.shape)}"
    )


def as_float32(image: torch.Tensor) -> torch.Tensor:
    if image.dtype == torch.float32:
        return image.contiguous()
    if image.dtype == torch.bool:
        raise ValueError("boolean images are not supported")
    # Integer inputs keep their numeric range. uint8 stays 0..255, matching
    # Mat::convertTo(..., CV_32F). Do not divide by 255.
    return image.to(dtype=torch.float32).contiguous()


def format_flow(flow: torch.Tensor, layout: str, as_numpy: bool) -> torch.Tensor | np.ndarray:
    if layout == "hw":
        out = flow[0].permute(1, 2, 0).contiguous()
        if as_numpy:
            return out.detach().cpu().numpy()
        return out
    if layout == "chw":
        return flow[0].contiguous()
    return flow.contiguous()


def prepare_pair(
    i0: Any,
    i1: Any,
    device: torch.device | None,
) -> tuple[torch.Tensor, torch.Tensor, str, bool]:
    """Validate a grayscale pair and return ``N,1,H,W`` float32 tensors."""
    as_numpy = is_numpy(i0)
    if as_numpy != is_numpy(i1):
        raise TypeError("I0 and I1 must both be NumPy arrays or both be torch tensors")
    if as_numpy:
        if i0.dtype != i1.dtype:
            raise ValueError(f"I0 and I1 dtypes differ: {i0.dtype} vs {i1.dtype}")
        if i0.shape != i1.shape:
            raise ValueError(f"I0 and I1 shapes differ: {i0.shape} vs {i1.shape}")
        left, layout = numpy_to_nchw(i0)
        right, _ = numpy_to_nchw(i1)
    else:
        if not isinstance(i0, torch.Tensor) or not isinstance(i1, torch.Tensor):
            raise TypeError("I0 and I1 must be NumPy arrays or torch tensors")
        if i0.dtype != i1.dtype:
            raise ValueError(f"I0 and I1 dtypes differ: {i0.dtype} vs {i1.dtype}")
        if i0.shape != i1.shape:
            raise ValueError(f"I0 and I1 shapes differ: {tuple(i0.shape)} vs {tuple(i1.shape)}")
        if i0.device != i1.device:
            raise ValueError(f"I0 and I1 devices differ: {i0.device} vs {i1.device}")
        left, layout = torch_to_nchw(i0)
        right, _ = torch_to_nchw(i1)

    compute = device
    if compute is None:
        compute = left.device if isinstance(i0, torch.Tensor) else torch.device("cpu")
    left = as_float32(left).to(compute)
    right = as_float32(right).to(compute)
    return left, right, layout, as_numpy


def validate_pyramid_settings(solver: Any) -> None:
    if not (0.0 < solver.downscale_factor < 1.0):
        raise ValueError(f"downscale_factor must be in (0, 1), got {solver.downscale_factor}")
    if solver.min_size < 1:
        raise ValueError(f"min_size must be >= 1, got {solver.min_size}")
    if solver.max_layers < 0:
        raise ValueError(f"max_layers must be >= 0, got {solver.max_layers}")
    if solver.fixed_point_iterations < 0 or solver.sor_iterations < 0:
        raise ValueError("iteration counts must be >= 0")
    if solver.epsilon < 0 or solver.zeta < 0:
        raise ValueError("zeta and epsilon must be >= 0")
