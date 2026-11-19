"""DeepFlow as described by Weinzaepfel et al., ICCV 2013.

Same pyramid and robust variational solver as :class:`deepflow.deepflow.DeepFlow`,
plus quasi-dense DeepMatching and a matching energy. Public smoothness, color,
and gradient weights are the paper's energy weights: this class does not apply
OpenCV's ``4 * alpha``, ``delta / 3``, and ``gamma / 3`` rescaling.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from deepflow import _xla
from deepflow.deepflow import _PyramidSolver
from deepflow.io import validate_pyramid_settings
from deepflow.matching import deep_match, resample_matches
from deepflow.ops import resize_bilinear
from deepflow.variational import variational_refine


def _reject_xla() -> None:
    # DeepMatching depends on conv2d precision (TPUs default to bfloat16 passes)
    # and on scatter_reduce tie-breaking; neither is validated on XLA yet.
    raise ValueError("DeepFlowICCV does not support TPU/XLA yet; use DeepFlow")


class DeepFlowICCV(_PyramidSolver):
    """ICCV 2013 DeepFlow: DeepMatching correspondences plus a matching term.

    Constructor fields shared with :class:`deepflow.deepflow.DeepFlow` keep that
    class's defaults. ``calc`` uses the same grayscale layouts. ``uint8`` stays
    on ``0..255``. Channel 0 of the flow is ``x`` (right) and channel 1 is ``y``
    (down). A preallocated ``flow`` argument is ignored.

    The matching weight at pyramid index ``k`` (``k = 0`` at the finest level,
    ``k_max`` at the coarsest) is ``beta * (k / k_max) ** beta_exponent``. A
    one-level pyramid uses ``beta`` unchanged.

    ``cuda_graph`` and ``compile`` behave as in :class:`deepflow.deepflow.DeepFlow`.
    The CUDA graph covers matching as well as the pyramid. Under ``compile``,
    the matching term can amplify one-ulp rounding changes more than DeepFlow does.
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
        beta: float = 300.0,
        beta_exponent: float = 0.6,
        sigma_m: float = 50.0,
        match_power: float = 1.4,
        descriptor_smooth: float = 1.0,
        sigmoid_slope: float = 0.2,
        descriptor_regularizer: float = 0.3,
        match_max_side: int | None = 256,
        cuda_graph: bool | None = None,
        compile: bool = False,
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
        if isinstance(device, str) and device.split(":")[0] == "tpu":
            _reject_xla()
        self.device = None if device is None else torch.device(device)
        if _xla.is_xla(self.device):
            _reject_xla()
        self.beta = float(beta)
        self.beta_exponent = float(beta_exponent)
        self.sigma_m = float(sigma_m)
        self.match_power = float(match_power)
        self.descriptor_smooth = float(descriptor_smooth)
        self.sigmoid_slope = float(sigmoid_slope)
        self.descriptor_regularizer = float(descriptor_regularizer)
        self.match_max_side = None if match_max_side is None else int(match_max_side)
        self._init_runtime(cuda_graph, compile)

    def calc(self, i0: Any, i1: Any, flow: Any = None) -> torch.Tensor | np.ndarray:
        """Estimate flow from ``i0`` to ``i1``. ``flow`` is accepted and ignored."""
        if self.device is None and any(
            isinstance(image, torch.Tensor) and _xla.is_xla(image.device) for image in (i0, i1)
        ):
            _reject_xla()
        return super().calc(i0, i1, flow)

    def _check_settings(self) -> None:
        validate_pyramid_settings(self)
        if self.beta < 0:
            raise ValueError(f"beta must be >= 0, got {self.beta}")
        if self.sigma_m <= 0:
            raise ValueError(f"sigma_m must be > 0, got {self.sigma_m}")
        if self.match_power <= 0:
            raise ValueError(f"match_power must be > 0, got {self.match_power}")
        if self.match_max_side is not None and self.match_max_side < 4:
            raise ValueError(f"match_max_side must be >= 4 or None, got {self.match_max_side}")

    def _settings(self) -> tuple[Any, ...]:
        return (
            self.sigma,
            self.min_size,
            self.downscale_factor,
            self.fixed_point_iterations,
            self.sor_iterations,
            self.alpha,
            self.delta,
            self.gamma,
            self.omega,
            self.max_layers,
            self.zeta,
            self.epsilon,
            self.beta,
            self.beta_exponent,
            self.sigma_m,
            self.match_power,
            self.descriptor_smooth,
            self.sigmoid_slope,
            self.descriptor_regularizer,
            self.match_max_side,
            self.compile,
        )

    def _estimate(self, i0: torch.Tensor, i1: torch.Tensor) -> torch.Tensor:
        matches = deep_match(
            i0,
            i1,
            sigma_m=self.sigma_m,
            match_power=self.match_power,
            descriptor_smooth=self.descriptor_smooth,
            sigmoid_slope=self.sigmoid_slope,
            descriptor_regularizer=self.descriptor_regularizer,
            match_max_side=self.match_max_side,
        )
        n = i0.shape[0]
        plan = self._plan(i0.shape[-2], i0.shape[-1], i0.device)
        pyramid = self._shared_pyramid(i0, i1, plan)
        coarsest = pyramid[-1][:n]
        flow_u = torch.zeros_like(coarsest)
        flow_v = torch.zeros_like(coarsest)
        scale = 1.0 / self.downscale_factor
        k_max = len(pyramid) - 1
        sor_step = self._sor_step()

        for level in range(k_max, -1, -1):
            level_h, level_w = plan.sizes[level]
            # k = 0 at the finest level. One level keeps beta as published.
            if k_max == 0:
                beta_k = self.beta
            else:
                beta_k = self.beta * (level / k_max) ** self.beta_exponent
            if beta_k == 0.0:
                match_u = match_v = match_weight = None
            else:
                match_u, match_v, match_weight = resample_matches(matches, level_h, level_w)
            flow_u, flow_v = variational_refine(
                pyramid[level][:n],
                pyramid[level][n:],
                flow_u,
                flow_v,
                alpha=self.alpha,
                delta=self.delta,
                gamma=self.gamma,
                omega=self.omega,
                fixed_point_iterations=self.fixed_point_iterations,
                sor_iterations=self.sor_iterations,
                zeta=self.zeta,
                epsilon=self.epsilon,
                match_u=match_u,
                match_v=match_v,
                match_weight=match_weight,
                beta=beta_k,
                masks=plan.masks[level],
                sor_step=sor_step,
            )
            if level > 0:
                next_size = plan.sizes[level - 1]
                flow_u = resize_bilinear(flow_u, next_size) * scale
                flow_v = resize_bilinear(flow_v, next_size) * scale

        return torch.cat([flow_u, flow_v], dim=1)


def create_opt_flow_deep_flow_iccv(**kwargs: Any) -> DeepFlowICCV:
    """Factory for :class:`DeepFlowICCV`."""
    return DeepFlowICCV(**kwargs)
