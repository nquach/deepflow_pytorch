"""Single-scale variational refinement, matched to ``cv::VariationalRefinement``.

The packed red/black buffers in OpenCV are a CPU SIMD layout. This module
implements the same stencil on dense ``N,1,H,W`` tensors so the update is
identical on CPU, CUDA, and MPS.
"""

from __future__ import annotations

import torch

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


def _add_horizontal(
    coeff: torch.Tensor, weight: torch.Tensor, delta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add an edge term onto both endpoints of every horizontal edge."""
    w = weight[..., :, :-1]
    left = torch.zeros_like(coeff)
    right = torch.zeros_like(coeff)
    left[..., :, :-1] = w
    right[..., :, 1:] = w
    b_left = torch.zeros_like(coeff)
    b_right = torch.zeros_like(coeff)
    b_left[..., :, :-1] = delta
    b_right[..., :, 1:] = -delta
    return coeff + left + right, b_left + b_right


def _add_vertical(
    coeff: torch.Tensor, weight: torch.Tensor, delta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    w = weight[..., :-1, :]
    top = torch.zeros_like(coeff)
    bottom = torch.zeros_like(coeff)
    top[..., :-1, :] = w
    bottom[..., 1:, :] = w
    b_top = torch.zeros_like(coeff)
    b_bottom = torch.zeros_like(coeff)
    b_top[..., :-1, :] = delta
    b_bottom[..., 1:, :] = -delta
    return coeff + top + bottom, b_top + b_bottom


def _apply_smoothness(
    a11: torch.Tensor,
    a22: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    weight: torch.Tensor,
    base_u: torch.Tensor,
    base_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Linear smoothness terms use the incoming flow, not the current increment."""
    du = base_u[..., :, 1:] - base_u[..., :, :-1]
    dv = base_v[..., :, 1:] - base_v[..., :, :-1]
    w = weight[..., :, :-1]
    a11, b1_h = _add_horizontal(a11, weight, w * du)
    a22, b2_h = _add_horizontal(a22, weight, w * dv)
    b1 = b1 + b1_h
    b2 = b2 + b2_h

    du = base_u[..., 1:, :] - base_u[..., :-1, :]
    dv = base_v[..., 1:, :] - base_v[..., :-1, :]
    w = weight[..., :-1, :]
    a11, b1_v = _add_vertical(a11, weight, w * du)
    a22, b2_v = _add_vertical(a22, weight, w * dv)
    return a11, a22, b1 + b1_v, b2 + b2_v


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


def _sigma(weight: torch.Tensor, inc: torch.Tensor) -> torch.Tensor:
    """Four-neighbor coupling. A missing neighbor contributes 0, not a replicate."""
    zeros_w = torch.zeros_like(inc[..., :, :1])
    zeros_h = torch.zeros_like(inc[..., :1, :])
    left = torch.cat([zeros_w, weight[..., :, :-1] * inc[..., :, :-1]], dim=-1)
    right = torch.cat([weight[..., :, :-1] * inc[..., :, 1:], zeros_w], dim=-1)
    up = torch.cat([zeros_h, weight[..., :-1, :] * inc[..., :-1, :]], dim=-2)
    down = torch.cat([weight[..., :-1, :] * inc[..., 1:, :], zeros_h], dim=-2)
    return left + right + up + down


def _checkerboard(height: int, width: int, device: torch.device, *, red: bool) -> torch.Tensor:
    ys = torch.arange(height, device=device)
    xs = torch.arange(width, device=device)
    mask = ((ys[:, None] + xs[None, :]) % 2) == 0
    if not red:
        mask = ~mask
    return mask.view(1, 1, height, width)


def _sor_pass(
    du: torch.Tensor,
    dv: torch.Tensor,
    weight: torch.Tensor,
    a11: torch.Tensor,
    a12: torch.Tensor,
    a22: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    omega: float,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sigma_u = _sigma(weight, du)
    sigma_v = _sigma(weight, dv)
    # dv at this pixel is updated from the just-computed du, matching the scalar loop.
    du_new = du + omega * ((sigma_u + b1 - dv * a12) / a11 - du)
    dv_new = dv + omega * ((sigma_v + b2 - du_new * a12) / a22 - dv)
    return torch.where(mask, du_new, du), torch.where(mask, dv_new, dv)


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

    du = torch.zeros_like(flow_u)
    dv = torch.zeros_like(flow_v)
    height, width = i0.shape[-2:]
    red = _checkerboard(height, width, i0.device, red=True)
    black = ~red

    for _ in range(fixed_point_iterations):
        a11, a12, a22, b1, b2 = _data_term(
            ix, iy, iz, ixx, ixy, iyy, ixz, iyz, du, dv, delta, gamma, zeta, epsilon
        )
        weight = _edge_weight(flow_u + du, flow_v + dv, alpha, epsilon)
        a11, a22, b1, b2 = _apply_smoothness(a11, a22, b1, b2, weight, flow_u, flow_v)
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
            du, dv = _sor_pass(du, dv, weight, a11, a12, a22, b1, b2, omega, red)
            du, dv = _sor_pass(du, dv, weight, a11, a12, a22, b1, b2, omega, black)

    return flow_u + du, flow_v + dv
