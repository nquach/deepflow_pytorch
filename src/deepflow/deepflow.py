"""Coarse-to-fine DeepFlow, matched to ``cv::optflow::createOptFlow_DeepFlow``."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from deepflow.io import format_flow, prepare_pair, validate_pyramid_settings
from deepflow.ops import gaussian_blur, resize_bilinear
from deepflow.variational import variational_refine


def build_pyramid(
    image: torch.Tensor,
    downscale_factor: float,
    min_size: int,
    max_layers: int,
) -> list[torch.Tensor]:
    """Successive ``INTER_LINEAR`` pyramid used by ``OpticalFlowDeepFlow::buildPyramid``.

    The next size on each axis is ``floor(prev * downscale_factor + 0.5)``. The
    loop stops when either side would be ``<= min_size``, and never exceeds
    ``max_layers`` extra resizes. Images are not blurred again between levels.
    """
    levels = [image]
    prev = image
    for _ in range(max_layers):
        height, width = prev.shape[-2:]
        next_h = int(math.floor(height * downscale_factor + 0.5))
        next_w = int(math.floor(width * downscale_factor + 0.5))
        if next_h <= min_size or next_w <= min_size:
            break
        prev = resize_bilinear(prev, (next_h, next_w))
        levels.append(prev)
    return levels


class DeepFlow:
    """Dense optical flow with the same defaults and pyramid as OpenCV DeepFlow.

    OpenCV's function is a coarse-to-fine wrapper around Brox variational
    refinement. It does not run the DeepMatching term from the ICCV 2013 paper.
    Public fields use OpenCV's names and values. Before the solver runs they are
    rescaled the same way ``deepflow.cpp`` does: smoothness ``4 * alpha``, color
    ``delta / 3``, gradient ``gamma / 3``.

    ``calc`` ignores any preallocated flow, matching OpenCV, which discards it
    and starts from zeros at the coarsest pyramid level.

    NumPy ``(H, W)`` or ``(H, W, 1)`` inputs return a float32 ``(H, W, 2)`` array
    (channel 0 is horizontal displacement in pixels, channel 1 is vertical, y
    pointing down). Torch inputs stay tensors on the compute device:

    - ``(H, W)`` and ``(1, H, W)`` return ``(2, H, W)``
    - ``(N, H, W)`` and ``(N, 1, H, W)`` return ``(N, 2, H, W)``

    A batch must share one spatial size. ``uint8`` is converted with
    ``float(value)``, not ``value / 255``. ``zeta`` and ``epsilon`` are absolute,
    so a ``[0, 1]`` float image is a different problem than an 8-bit image.
    """

    def __init__(
        self,
        sigma: float = 0.6,
        min_size: int = 25,
        downscale_factor: float = 0.95,
        fixed_point_iterations: int = 5,
        sor_iterations: int = 25,
        alpha: float = 1.0,
        delta: float = 0.5,
        gamma: float = 5.0,
        omega: float = 1.6,
        max_layers: int = 200,
        zeta: float = 0.1,
        epsilon: float = 0.001,
        device: str | torch.device | None = None,
    ) -> None:
        self.sigma = float(sigma)
        self.min_size = int(min_size)
        self.downscale_factor = float(downscale_factor)
        self.fixed_point_iterations = int(fixed_point_iterations)
        self.sor_iterations = int(sor_iterations)
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.gamma = float(gamma)
        self.omega = float(omega)
        self.max_layers = int(max_layers)
        self.zeta = float(zeta)
        self.epsilon = float(epsilon)
        self.device = None if device is None else torch.device(device)

    def calc(self, i0: Any, i1: Any, flow: Any = None) -> torch.Tensor | np.ndarray:
        """Estimate flow from ``i0`` to ``i1``. ``flow`` is accepted and ignored."""
        del flow
        self._check_settings()
        left, right, layout, as_numpy = prepare_pair(i0, i1, self.device)
        with torch.no_grad():
            flow_nchw = self._estimate(left, right)
        return format_flow(flow_nchw, layout, as_numpy)

    def _check_settings(self) -> None:
        validate_pyramid_settings(self)

    def _estimate(self, i0: torch.Tensor, i1: torch.Tensor) -> torch.Tensor:
        i0 = gaussian_blur(i0, self.sigma)
        i1 = gaussian_blur(i1, self.sigma)
        pyramid0 = build_pyramid(i0, self.downscale_factor, self.min_size, self.max_layers)
        pyramid1 = build_pyramid(i1, self.downscale_factor, self.min_size, self.max_layers)
        coarsest = pyramid0[-1]
        flow_u = torch.zeros_like(coarsest)
        flow_v = torch.zeros_like(coarsest)
        scale = 1.0 / self.downscale_factor
        alpha = 4.0 * self.alpha
        delta = self.delta / 3.0
        gamma = self.gamma / 3.0

        for level in range(len(pyramid0) - 1, -1, -1):
            flow_u, flow_v = variational_refine(
                pyramid0[level],
                pyramid1[level],
                flow_u,
                flow_v,
                alpha=alpha,
                delta=delta,
                gamma=gamma,
                omega=self.omega,
                fixed_point_iterations=self.fixed_point_iterations,
                sor_iterations=self.sor_iterations,
                zeta=self.zeta,
                epsilon=self.epsilon,
            )
            if level > 0:
                next_size = pyramid0[level - 1].shape[-2:]
                flow_u = resize_bilinear(flow_u, next_size) * scale
                flow_v = resize_bilinear(flow_v, next_size) * scale

        return torch.cat([flow_u, flow_v], dim=1)


def create_opt_flow_deep_flow(**kwargs: Any) -> DeepFlow:
    """Factory with the same role as ``cv2.optflow.createOptFlow_DeepFlow``."""
    return DeepFlow(**kwargs)
