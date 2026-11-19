"""Coarse-to-fine DeepFlow, matched to ``cv::optflow::createOptFlow_DeepFlow``."""

from __future__ import annotations

import contextlib
import functools
import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from deepflow import _xla
from deepflow.io import format_flow, prepare_pair, resolve_device, validate_pyramid_settings
from deepflow.ops import gaussian_blur, resize_bilinear
from deepflow.variational import (
    checkerboard,
    fixed_point_step,
    sor_iteration_functional,
    variational_refine,
)


def pyramid_sizes(
    height: int,
    width: int,
    downscale_factor: float,
    min_size: int,
    max_layers: int,
) -> list[tuple[int, int]]:
    """``(height, width)`` of every level of :func:`build_pyramid`, finest first."""
    sizes = [(height, width)]
    for _ in range(max_layers):
        prev_h, prev_w = sizes[-1]
        next_h = int(math.floor(prev_h * downscale_factor + 0.5))
        next_w = int(math.floor(prev_w * downscale_factor + 0.5))
        if next_h <= min_size or next_w <= min_size:
            break
        sizes.append((next_h, next_w))
    return sizes


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
    height, width = image.shape[-2:]
    levels = [image]
    for size in pyramid_sizes(height, width, downscale_factor, min_size, max_layers)[1:]:
        levels.append(resize_bilinear(levels[-1], size))
    return levels


@dataclass
class _Plan:
    """Per-size constants, built once so ``_estimate`` does no host-to-device copy."""

    sizes: list[tuple[int, int]]
    masks: list[tuple[torch.Tensor, torch.Tensor]]


@dataclass
class _GraphEntry:
    graph: torch.cuda.CUDAGraph
    static_i0: torch.Tensor
    static_i1: torch.Tensor
    static_flow: torch.Tensor


_COMPILED_SOR: Callable[..., None] | None = None


def _compiled_sor_iteration() -> Callable[..., None]:
    global _COMPILED_SOR
    if _COMPILED_SOR is None:
        # dynamic=True: one compiled graph serves every pyramid level size.
        _COMPILED_SOR = torch.compile(sor_iteration_functional, dynamic=True, fullgraph=True)
    return _COMPILED_SOR


def _deepflow_level(
    i0: torch.Tensor,
    i1: torch.Tensor,
    flow_u: torch.Tensor,
    flow_v: torch.Tensor,
    red: torch.Tensor,
    black: torch.Tensor,
    *,
    alpha: float,
    delta: float,
    gamma: float,
    omega: float,
    fixed_point_iterations: int,
    sor_iterations: int,
    zeta: float,
    epsilon: float,
    sor_step: Callable[..., None] | None = None,
    fixed_point_step_fn: Callable[..., None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refine one pyramid level (no upsampling)."""
    return variational_refine(
        i0,
        i1,
        flow_u,
        flow_v,
        alpha=alpha,
        delta=delta,
        gamma=gamma,
        omega=omega,
        fixed_point_iterations=fixed_point_iterations,
        sor_iterations=sor_iterations,
        zeta=zeta,
        epsilon=epsilon,
        masks=(red, black),
        sor_step=sor_step,
        fixed_point_step_fn=fixed_point_step_fn,
    )


def _deepflow_level_functional(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`_deepflow_level` with the out-of-place SOR step, which XLA traces cleanly."""
    return _deepflow_level(*args, sor_step=sor_iteration_functional, **kwargs)


def _fixed_point_step_functional(*args: Any, **kwargs: Any) -> None:
    return fixed_point_step(*args, sor_step=sor_iteration_functional, **kwargs)


_XLA_COMPILED_STEP: Callable[..., None] | None = None


def _xla_compiled_level(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`_deepflow_level_functional` with each fixed-point step run through openxla.

    The compiled unit is one fixed-point step (25 SOR iterations), not a whole
    level: a level reuses it ``fixed_point_iterations`` times with the same
    shapes, so Dynamo traces a fifth as many ops. Static shapes, so XLA builds
    one executable per level size and later frames replay it without retracing.
    """
    global _XLA_COMPILED_STEP
    if _XLA_COMPILED_STEP is None:
        _XLA_COMPILED_STEP = torch.compile(_fixed_point_step_functional, backend="openxla", dynamic=False)
    return _deepflow_level_functional(*args, fixed_point_step_fn=_XLA_COMPILED_STEP, **kwargs)


_RECOMPILE_LIMIT = 2048


def _xla_recompile_headroom() -> contextlib.AbstractContextManager[Any]:
    """Raise Dynamo's per-function recompile limit while the level loop runs.

    Every pyramid level size is a separate entry (about 70 per resolution), well
    above the default limit of 8. Names differ across torch releases.
    """
    import torch._dynamo

    config = torch._dynamo.config
    names = ("recompile_limit", "cache_size_limit", "accumulated_recompile_limit", "accumulated_cache_size_limit")
    changes = {name: _RECOMPILE_LIMIT for name in names if hasattr(config, name)}
    return config.patch(**changes)


class _PyramidSolver:
    """Shared runtime for the coarse-to-fine solvers.

    Subclasses set the public fields, call ``_init_runtime``, and implement
    ``_check_settings``, ``_settings`` (every field that changes the result), and
    ``_estimate(i0, i1) -> (N, 2, H, W)``. ``_estimate`` must not sync with the
    host or copy from it once the per-shape caches are warm, so it can be
    recorded as a CUDA graph.
    """

    downscale_factor: float
    min_size: int
    max_layers: int
    sigma: float
    device: torch.device | None

    def _init_runtime(self, cuda_graph: bool | None, compile: bool) -> None:
        self.cuda_graph = cuda_graph
        self.compile = bool(compile)
        self._plans: dict[tuple[Any, ...], _Plan] = {}
        self._graphs: dict[tuple[Any, ...], _GraphEntry] = {}
        self._graph_failed = False

    def calc(self, i0: Any, i1: Any, flow: Any = None) -> torch.Tensor | np.ndarray:
        """Estimate flow from ``i0`` to ``i1``. ``flow`` is accepted and ignored."""
        del flow
        self._check_settings()
        left, right, layout, as_numpy = prepare_pair(i0, i1, self.device)
        with torch.no_grad():
            if self._use_graph(left.device):
                flow_nchw = self._estimate_graph(left, right)
            else:
                flow_nchw = self._estimate(left, right)
        return format_flow(flow_nchw, layout, as_numpy)

    def _check_settings(self) -> None:
        raise NotImplementedError

    def _settings(self) -> tuple[Any, ...]:
        raise NotImplementedError

    def _estimate(self, i0: torch.Tensor, i1: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _use_graph(self, device: torch.device) -> bool:
        if device.type != "cuda" or self._graph_failed:
            return False
        return self.cuda_graph is None or bool(self.cuda_graph)

    def _sor_step(self) -> Callable[..., None] | None:
        return _compiled_sor_iteration() if self.compile else None

    def _plan(self, height: int, width: int, device: torch.device) -> _Plan:
        key = (height, width, device, self.downscale_factor, self.min_size, self.max_layers)
        plan = self._plans.get(key)
        if plan is None:
            sizes = pyramid_sizes(height, width, self.downscale_factor, self.min_size, self.max_layers)
            masks = [checkerboard(h, w, device) for h, w in sizes]
            plan = _Plan(sizes, masks)
            self._plans[key] = plan
        return plan

    def _shared_pyramid(self, i0: torch.Tensor, i1: torch.Tensor, plan: _Plan) -> list[torch.Tensor]:
        """Blurred pyramid of ``cat([i0, i1])``: one blur and one resize per level for both frames."""
        level = gaussian_blur(torch.cat([i0, i1], dim=0), self.sigma)
        pyramid = [level]
        for size in plan.sizes[1:]:
            level = resize_bilinear(level, size)
            pyramid.append(level)
        return pyramid

    def _estimate_graph(self, i0: torch.Tensor, i1: torch.Tensor) -> torch.Tensor:
        key = (tuple(i0.shape), i0.device, self._settings())
        entry = self._graphs.get(key)
        if entry is None:
            try:
                entry = self._capture(i0, i1)
            except Exception as error:  # noqa: BLE001 - any capture failure falls back
                self._graph_failed = True
                warnings.warn(
                    f"CUDA graph capture failed ({error}); {type(self).__name__} falls back to eager mode.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return self._estimate(i0, i1)
            self._graphs[key] = entry
        else:
            entry.static_i0.copy_(i0)
            entry.static_i1.copy_(i1)
        entry.graph.replay()
        return entry.static_flow.clone()

    def _capture(self, i0: torch.Tensor, i1: torch.Tensor) -> _GraphEntry:
        static_i0 = i0.clone()
        static_i1 = i1.clone()
        # Warm up on a side stream: fills the per-shape caches (and compiles, if
        # enabled) so nothing inside the capture allocates on or copies from the host.
        side = torch.cuda.Stream(device=i0.device)
        side.wait_stream(torch.cuda.current_stream(i0.device))
        with torch.cuda.stream(side):
            self._estimate(static_i0, static_i1)
        torch.cuda.current_stream(i0.device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_flow = self._estimate(static_i0, static_i1)
        return _GraphEntry(graph, static_i0, static_i1, static_flow)


class DeepFlow(_PyramidSolver):
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

    Speed options (results are unchanged unless noted):

    - ``cuda_graph``: on CUDA, the first call for each input shape records the
      whole pyramid as one CUDA graph; later calls replay it. ``None`` (default)
      enables it on CUDA devices. Each recorded shape keeps its own GPU memory.
    - ``compile``: run the SOR iteration through ``torch.compile``. The first
      call is slow, and fused kernels may round differently by about one ulp.
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
        self.device = resolve_device(device)
        self._init_runtime(cuda_graph, compile)

    def _check_settings(self) -> None:
        validate_pyramid_settings(self)

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
            self.compile,
        )

    def _estimate(self, i0: torch.Tensor, i1: torch.Tensor) -> torch.Tensor:
        n = i0.shape[0]
        plan = self._plan(i0.shape[-2], i0.shape[-1], i0.device)
        pyramid = self._shared_pyramid(i0, i1, plan)
        coarsest = pyramid[-1][:n]
        flow_u = torch.zeros_like(coarsest)
        flow_v = torch.zeros_like(coarsest)
        scale = 1.0 / self.downscale_factor
        alpha = 4.0 * self.alpha
        delta = self.delta / 3.0
        gamma = self.gamma / 3.0
        params = dict(
            alpha=alpha,
            delta=delta,
            gamma=gamma,
            omega=self.omega,
            fixed_point_iterations=self.fixed_point_iterations,
            sor_iterations=self.sor_iterations,
            zeta=self.zeta,
            epsilon=self.epsilon,
        )
        on_xla = _xla.is_xla(i0.device)
        if on_xla and self.compile:
            level_fn = _xla_compiled_level
        elif on_xla:
            level_fn = _deepflow_level_functional
        else:
            level_fn = functools.partial(_deepflow_level, sor_step=self._sor_step())

        with _xla_recompile_headroom() if on_xla and self.compile else contextlib.nullcontext():
            for index in range(len(pyramid) - 1, -1, -1):
                red, black = plan.masks[index]
                flow_u, flow_v = level_fn(
                    pyramid[index][:n], pyramid[index][n:], flow_u, flow_v, red, black, **params
                )
                if index > 0:
                    next_size = plan.sizes[index - 1]
                    flow_u = resize_bilinear(flow_u, next_size) * scale
                    flow_v = resize_bilinear(flow_v, next_size) * scale
                if on_xla:
                    # One lazy graph per level, cached by shape, instead of one for the whole frame.
                    _xla.sync()

        return torch.cat([flow_u, flow_v], dim=1)


def create_opt_flow_deep_flow(**kwargs: Any) -> DeepFlow:
    """Factory with the same role as ``cv2.optflow.createOptFlow_DeepFlow``."""
    return DeepFlow(**kwargs)
