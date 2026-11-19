"""Single-scale variational refinement, matched to ``cv::VariationalRefinement``.

The packed red/black buffers in OpenCV are a CPU SIMD layout. This module
implements the same stencil on dense ``N,1,H,W`` tensors so the update is
identical on CPU, CUDA, and MPS.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from deepflow.ops import remap_bilinear, sobel_x, sobel_y


def _edge_weight(flow_u: torch.Tensor, flow_v: torch.Tensor, alpha: float, epsilon: float) -> torch.Tensor:
    """Robust smoothness weight owned by each pixel's right and down edges.

    Missing neighbors are treated as replicates of the border sample, so the
    difference across a nonexistent edge is zero. That matches the repeated
    borders OpenCV writes into the checkerboard buffers.
    """
    u_right = torch.cat([flow_u[..., :, 1:], flow_u[..., :, -1:]], dim=-1)
    v_right = torch.cat([flow_v[..., :, 1:], flow_v[..., :, -1:]], dim=-1)
    u_down = torch.cat([flow_u[..., 1:, :], flow_u[..., -1:, :]], dim=-2)
    v_down = torch.cat([flow_v[..., 1:, :], flow_v[..., -1:, :]], dim=-2)
    ux = u_right - flow_u
    vx = v_right - flow_v
    uy = u_down - flow_u
    vy = v_down - flow_v
    alpha2 = alpha / 2.0
    eps2 = epsilon * epsilon
    return alpha2 / torch.sqrt(ux * ux + vx * vx + uy * uy + vy * vy + eps2)


def _neighbor_weights(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Edge weight toward each of the four neighbors, 0 where the neighbor is missing.

    Returned as full-size ``(left, right, up, down)`` maps. ``right`` and ``down``
    are the pixel's own edges; ``left`` and ``up`` are the neighbor's edges.
    """
    wh = weight[..., :, :-1]
    wv = weight[..., :-1, :]
    return (
        F.pad(wh, (1, 0)),
        F.pad(wh, (0, 1)),
        F.pad(wv, (0, 0, 1, 0)),
        F.pad(wv, (0, 0, 0, 1)),
    )


def _edge_rhs(delta: torch.Tensor, *, horizontal: bool) -> torch.Tensor:
    """``+delta`` at the edge's first endpoint and ``-delta`` at its second."""
    if horizontal:
        return F.pad(delta, (0, 1)) + F.pad(-delta, (1, 0))
    return F.pad(delta, (0, 0, 0, 1)) + F.pad(-delta, (0, 0, 1, 0))


def _apply_smoothness(
    a11: torch.Tensor,
    a22: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    weight: torch.Tensor,
    neighbors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    base_u: torch.Tensor,
    base_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Linear smoothness terms use the incoming flow, not the current increment."""
    w_left, w_right, w_up, w_down = neighbors
    w = weight[..., :, :-1]
    a11 = a11 + w_right + w_left
    a22 = a22 + w_right + w_left
    b1 = b1 + _edge_rhs(w * (base_u[..., :, 1:] - base_u[..., :, :-1]), horizontal=True)
    b2 = b2 + _edge_rhs(w * (base_v[..., :, 1:] - base_v[..., :, :-1]), horizontal=True)

    w = weight[..., :-1, :]
    a11 = a11 + w_down + w_up
    a22 = a22 + w_down + w_up
    b1 = b1 + _edge_rhs(w * (base_u[..., 1:, :] - base_u[..., :-1, :]), horizontal=False)
    b2 = b2 + _edge_rhs(w * (base_v[..., 1:, :] - base_v[..., :-1, :]), horizontal=False)
    return a11, a22, b1, b2


def _data_term(
    ix: torch.Tensor,
    iy: torch.Tensor,
    iz: torch.Tensor,
    ixx: torch.Tensor,
    ixy: torch.Tensor,
    iyy: torch.Tensor,
    ixz: torch.Tensor,
    iyz: torch.Tensor,
    du: torch.Tensor,
    dv: torch.Tensor,
    delta: float,
    gamma: float,
    zeta: float,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Color and gradient constancy, following ``ComputeDataTerm_ParBody``."""
    zeta2 = zeta * zeta
    eps2 = epsilon * epsilon
    delta2 = delta / 2.0
    gamma2 = gamma / 2.0

    deriv = ix * ix + iy * iy + zeta2
    ik1z = iz + ix * du + iy * dv
    weight = (delta2 / torch.sqrt(ik1z * ik1z / deriv + eps2)) / deriv
    a11 = weight * (ix * ix) + zeta2
    a12 = weight * (ix * iy)
    a22 = weight * (iy * iy) + zeta2
    b1 = -weight * (iz * ix)
    b2 = -weight * (iz * iy)

    deriv_x = ixx * ixx + ixy * ixy + zeta2
    deriv_y = iyy * iyy + ixy * ixy + zeta2
    ik1zx = ixz + ixx * du + ixy * dv
    ik1zy = iyz + ixy * du + iyy * dv
    weight = gamma2 / torch.sqrt(ik1zx * ik1zx / deriv_x + ik1zy * ik1zy / deriv_y + eps2)
    a11 = a11 + weight * (ixx * ixx / deriv_x + ixy * ixy / deriv_y)
    a12 = a12 + weight * (ixx * ixy / deriv_x + ixy * iyy / deriv_y)
    a22 = a22 + weight * (ixy * ixy / deriv_x + iyy * iyy / deriv_y)
    b1 = b1 - weight * (ixx * ixz / deriv_x + ixy * iyz / deriv_y)
    b2 = b2 - weight * (ixy * ixz / deriv_x + iyy * iyz / deriv_y)
    return a11, a12, a22, b1, b2


def _sigma(
    neighbors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], inc: torch.Tensor
) -> torch.Tensor:
    """Four-neighbor coupling. A missing neighbor contributes 0, not a replicate.

    ``inc`` may stack several channels (``du`` and ``dv``); the weights broadcast.
    """
    w_left, w_right, w_up, w_down = neighbors
    padded = F.pad(inc, (1, 1, 1, 1))
    # Separate multiply and add (not addcmul) keep the rounding of the scalar loop.
    total = w_left * padded[..., 1:-1, :-2]
    total = total + w_right * padded[..., 1:-1, 2:]
    total = total + w_up * padded[..., :-2, 1:-1]
    return total + w_down * padded[..., 2:, 1:-1]


def checkerboard(height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """``(red, black)`` masks shaped ``1,1,H,W``. Red is ``(y + x)`` even."""
    ys = torch.arange(height, device=device)
    xs = torch.arange(width, device=device)
    red = (((ys[:, None] + xs[None, :]) % 2) == 0).view(1, 1, height, width)
    return red, ~red


def _sor_pass(
    inc: torch.Tensor,
    neighbors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    a11: torch.Tensor,
    a12: torch.Tensor,
    a22: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    omega: float,
    mask: torch.Tensor,
) -> None:
    """Update the pixels in ``mask`` of the stacked ``inc = (du, dv)`` in place."""
    sigma = _sigma(neighbors, inc)
    du = inc[:, 0:1]
    dv = inc[:, 1:2]
    du_new = du + omega * ((sigma[:, 0:1] + b1 - dv * a12) / a11 - du)
    torch.where(mask, du_new, du, out=du)
    # dv at this pixel reads the just-updated du, matching the scalar loop.
    dv_new = dv + omega * ((sigma[:, 1:2] + b2 - du * a12) / a22 - dv)
    torch.where(mask, dv_new, dv, out=dv)


def sor_iteration(
    inc: torch.Tensor,
    neighbors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    a11: torch.Tensor,
    a12: torch.Tensor,
    a22: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    omega: float,
    red: torch.Tensor,
    black: torch.Tensor,
) -> None:
    """One red pass then one black pass, in place on ``inc``."""
    _sor_pass(inc, neighbors, a11, a12, a22, b1, b2, omega, red)
    _sor_pass(inc, neighbors, a11, a12, a22, b1, b2, omega, black)


def warp_image(image: torch.Tensor, flow_u: torch.Tensor, flow_v: torch.Tensor) -> torch.Tensor:
    """Warp ``image`` by ``(x + u, y + v)`` with bilinear sampling and replicate borders."""
    height, width = image.shape[-2:]
    xs = torch.arange(width, device=image.device, dtype=image.dtype).view(1, 1, 1, width)
    ys = torch.arange(height, device=image.device, dtype=image.dtype).view(1, 1, height, 1)
    return remap_bilinear(image, xs + flow_u, ys + flow_v)


def variational_refine(
    i0: torch.Tensor,
    i1: torch.Tensor,
    flow_u: torch.Tensor,
    flow_v: torch.Tensor,
    *,
    alpha: float,
    delta: float,
    gamma: float,
    omega: float = 1.6,
    fixed_point_iterations: int = 5,
    sor_iterations: int = 5,
    zeta: float = 0.1,
    epsilon: float = 0.001,
    match_u: torch.Tensor | None = None,
    match_v: torch.Tensor | None = None,
    match_weight: torch.Tensor | None = None,
    beta: float = 0.0,
    masks: tuple[torch.Tensor, torch.Tensor] | None = None,
    sor_step: Callable[..., None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refine ``flow_u`` / ``flow_v`` at a single resolution.

    Images and flow are ``N,1,H,W`` float tensors. Derivatives are taken once
    from ``i1`` warped by the incoming flow. Fixed-point iterations update the
    robust weights and the flow increment, and do not warp again. ``alpha``,
    ``delta``, and ``gamma`` are the weights OpenCV's refinement class stores
    (DeepFlow rescales its public fields before calling this).

    ``match_u``, ``match_v``, and ``match_weight`` are optional. When they are
    omitted, or ``beta`` is 0, the coefficients are exactly the color, gradient,
    and smoothness terms above. ``match_weight`` is the per-pixel mask times the
    match confidence.

    ``masks`` is an optional precomputed ``checkerboard`` pair for this size.
    ``sor_step`` replaces :func:`sor_iteration` (for example a compiled copy)
    and must update ``inc`` in place.
    """
    if i0.shape != i1.shape:
        raise ValueError(f"I0 and I1 shapes differ: {tuple(i0.shape)} vs {tuple(i1.shape)}")
    if flow_u.shape != i0.shape or flow_v.shape != i0.shape:
        raise ValueError("flow must have the same shape as the images")

    warped = warp_image(i1, flow_u, flow_v)
    iz = warped - i0
    averaged = 0.5 * i0 + 0.5 * warped
    ix = sobel_x(averaged)
    iy = sobel_y(averaged)
    ixz = sobel_x(iz)
    iyz = sobel_y(iz)
    ixx = sobel_x(ix)
    ixy = sobel_y(ix)
    iyy = sobel_y(iy)

    n, _, height, width = flow_u.shape
    inc = flow_u.new_zeros((n, 2, height, width))
    du = inc[:, 0:1]
    dv = inc[:, 1:2]
    red, black = masks if masks is not None else checkerboard(height, width, i0.device)
    step = sor_iteration if sor_step is None else sor_step

    for _ in range(fixed_point_iterations):
        a11, a12, a22, b1, b2 = _data_term(
            ix, iy, iz, ixx, ixy, iyy, ixz, iyz, du, dv, delta, gamma, zeta, epsilon
        )
        weight = _edge_weight(flow_u + du, flow_v + dv, alpha, epsilon)
        neighbors = _neighbor_weights(weight)
        a11, a22, b1, b2 = _apply_smoothness(a11, a22, b1, b2, weight, neighbors, flow_u, flow_v)
        if match_u is not None and beta != 0.0:
            # Weight uses the current flow W + dW. The linear pull is toward the
            # match from the incoming flow W, same fixed-point split as color.
            diff_u = flow_u + du - match_u
            diff_v = flow_v + dv - match_v
            rho = (beta * 0.5) * match_weight / torch.sqrt(
                diff_u * diff_u + diff_v * diff_v + epsilon * epsilon
            )
            a11 = a11 + rho
            a22 = a22 + rho
            b1 = b1 + rho * (match_u - flow_u)
            b2 = b2 + rho * (match_v - flow_v)
        for _ in range(sor_iterations):
            step(inc, neighbors, a11, a12, a22, b1, b2, omega, red, black)

    return flow_u + du, flow_v + dv
