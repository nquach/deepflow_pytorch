"""Quasi-dense DeepMatching, without the fast k-means approximation.

The 2D filters, aggregation, and top-level backtracking follow the extended
description in Revaud et al., arXiv:1506.07656. Entry points are every location
of the top correlation map. The match confidence uses the ICCV formula, without
the later square-root on the structure-tensor eigenvalue. Scale and rotation
search are not implemented.

Reference patches are correlated in chunks. Only the subsampled map and the
3x3 argmax offset are kept, so the full-resolution correlation volume is never
stored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from deepflow.ops import gaussian_blur, remap_bilinear, resize_bilinear, sobel_x, sobel_y

_CHUNK = 16
_SENTINEL = -1.0e6
# Correlation scores from conv and max differ by about one float32 ulp across
# devices. Treat values this close as ties and keep a fixed winner.
_TIE = 1.0e-5
# Quadrant offsets o_i. Aggregation reads the child at p' - o (the paper's T_o).
_QUADRANTS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


@dataclass
class MatchField:
    """Correspondences at the resolution where matching ran.

    ``u`` and ``v`` are displacements in pixels of that image. ``mask`` is 1
    where a reciprocal match was kept. ``phi`` is the ICCV confidence, already
    0 where ``mask`` is 0.
    """

    u: torch.Tensor
    v: torch.Tensor
    mask: torch.Tensor
    phi: torch.Tensor


@dataclass
class _Node:
    pooled: torch.Tensor | None
    argmax_y: torch.Tensor | None
    argmax_x: torch.Tensor | None
    score: torch.Tensor | None
    gy: int
    gx: int


def pixel_descriptors(
    image: torch.Tensor,
    descriptor_smooth: float,
    sigmoid_slope: float,
    descriptor_regularizer: float,
) -> torch.Tensor:
    """9-bin pixel descriptor. ``image`` is ``N,1,H,W`` float.

    Smooth, take the replicate central-difference gradient, and project onto
    the eight orientations ``i * pi / 4`` with a non-negative projection.
    Smooth, apply the sigmoid, smooth again, append the constant bin, and
    L2-normalize. Dot products of the result lie in ``[0, 1]``.
    """
    blurred = gaussian_blur(image, descriptor_smooth)
    gx = sobel_x(blurred)
    gy = sobel_y(blurred)
    angles = torch.arange(1, 9, device=image.device, dtype=image.dtype) * (math.pi / 4.0)
    cosine = torch.cos(angles).view(1, 8, 1, 1)
    sine = torch.sin(angles).view(1, 8, 1, 1)
    projected = (gx * cosine + gy * sine).clamp(min=0)
    projected = gaussian_blur(projected, descriptor_smooth)
    projected = 2.0 / (1.0 + torch.exp(-sigmoid_slope * projected)) - 1.0
    projected = gaussian_blur(projected, descriptor_smooth)
    regularizer = torch.full(
        (image.shape[0], 1, image.shape[2], image.shape[3]),
        float(descriptor_regularizer),
        dtype=image.dtype,
        device=image.device,
    )
    descriptor = torch.cat([projected, regularizer], dim=1)
    norm = descriptor.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return descriptor / norm


def average_and_power(
    children: list[torch.Tensor],
    valid: list[torch.Tensor | bool],
    power: float,
) -> torch.Tensor:
    """Mean of the valid children, then raise to ``power``.

    A missing child is left out of both the sum and the count, including when
    it is missing only on the border of a shifted map.
    """
    total = torch.zeros_like(children[0])
    count = torch.zeros_like(children[0])
    for child, ok in zip(children, valid):
        if isinstance(ok, torch.Tensor):
            mask = ok.to(dtype=child.dtype)
        elif ok:
            mask = torch.ones_like(child)
        else:
            mask = torch.zeros_like(child)
        total = total + child * mask
        count = count + mask
    mean = total / count.clamp(min=1)
    powered = mean.clamp(min=0).pow(power)
    return torch.where(count > 0, powered, torch.zeros_like(powered))


def aggregate_pooled(pooled: torch.Tensor, power: float) -> torch.Tensor:
    """One aggregation step on a ``(gy, gx, H, W)`` map of child scores.

    Each parent averages the four children after the quadrant shift. A child
    that falls outside the map is dropped and the average is renormalized.
    """
    gy, gx, _, _ = pooled.shape
    if gy < 2 or gx < 2:
        raise ValueError("aggregation needs at least a 2x2 patch grid")
    shifted: list[torch.Tensor] = []
    valid: list[torch.Tensor] = []
    for oy, ox in _QUADRANTS:
        iy0 = 0 if oy < 0 else 1
        ix0 = 0 if ox < 0 else 1
        child = pooled[iy0 : iy0 + gy - 1, ix0 : ix0 + gx - 1]
        moved, mask = _shift_with_mask(child, oy, ox)
        shifted.append(moved)
        valid.append(mask)
    return average_and_power(shifted, valid, power)


def atomic_correlation(
    reference: torch.Tensor,
    target: torch.Tensor,
    *,
    descriptor_smooth: float = 1.0,
    sigmoid_slope: float = 0.2,
    descriptor_regularizer: float = 0.3,
) -> torch.Tensor:
    """Bottom-level correlation, shape ``(gy, gx, H, W)``, before subsampling.

    Patch ``(iy, ix)`` is the 4x4 window centered on the reference grid
    ``{2, 6, ...}``. Value ``(y, x)`` is the mean descriptor dot product with
    the target window centered at ``(y, x)``.
    """
    ref = _as_one(reference)
    tgt = _as_one(target)
    desc_ref = pixel_descriptors(ref, descriptor_smooth, sigmoid_slope, descriptor_regularizer)
    desc_tgt = pixel_descriptors(tgt, descriptor_smooth, sigmoid_slope, descriptor_regularizer)
    centers_y = _centers(ref.shape[-2], ref.device)
    centers_x = _centers(ref.shape[-1], ref.device)
    if centers_y.numel() == 0 or centers_x.numel() == 0:
        return ref.new_zeros((0, 0, ref.shape[-2], ref.shape[-1]))
    patches = _reference_patches(desc_ref, centers_y, centers_x)
    response = _correlate_chunks(desc_tgt, patches)
    return response.view(centers_y.numel(), centers_x.numel(), ref.shape[-2], ref.shape[-1])


def deep_match(
    image0: torch.Tensor,
    image1: torch.Tensor,
    *,
    sigma_m: float = 50.0,
    match_power: float = 1.4,
    descriptor_smooth: float = 1.0,
    sigmoid_slope: float = 0.2,
    descriptor_regularizer: float = 0.3,
    match_max_side: int | None = 256,
) -> MatchField:
    """Match a grayscale pair. Images are ``N,1,H,W`` float.

    When ``match_max_side`` is set, both images are resized so the long side
    is at most that length, preserving aspect ratio. Displacements are returned
    in that resized pixel grid.
    """
    if image0.ndim != 4 or image0.shape[1] != 1:
        raise ValueError(f"expected N,1,H,W images, got {tuple(image0.shape)}")
    if image0.shape != image1.shape:
        raise ValueError(f"image shapes differ: {tuple(image0.shape)} vs {tuple(image1.shape)}")
    height, width = image0.shape[-2:]
    match_h, match_w = _match_size(height, width, match_max_side)
    if (match_h, match_w) != (height, width):
        image0 = resize_bilinear(image0, (match_h, match_w))
        image1 = resize_bilinear(image1, (match_h, match_w))
    fields = [
        _match_one(
            image0[index : index + 1],
            image1[index : index + 1],
            sigma_m=sigma_m,
            match_power=match_power,
            descriptor_smooth=descriptor_smooth,
            sigmoid_slope=sigmoid_slope,
            descriptor_regularizer=descriptor_regularizer,
        )
        for index in range(image0.shape[0])
    ]
    return MatchField(
        u=torch.cat([field.u for field in fields], dim=0),
        v=torch.cat([field.v for field in fields], dim=0),
        mask=torch.cat([field.mask for field in fields], dim=0),
        phi=torch.cat([field.phi for field in fields], dim=0),
    )


def resample_matches(
    field: MatchField,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bring matches onto a flow-pyramid level.

    Displacements and the mask are nearest-neighbor, then displacements are
    multiplied by the true size ratio of this level to the matching resolution.
    Confidence is bilinear and multiplied by the resized mask.
    """
    source_h, source_w = field.u.shape[-2:]
    if (source_h, source_w) == (height, width):
        u, v, mask, phi = field.u, field.v, field.mask, field.phi
    else:
        size = (height, width)
        u = resize_bilinear(field.u, size)
        v = resize_bilinear(field.v, size)
        mask = F.interpolate(field.mask, size=size, mode="nearest")
        phi = resize_bilinear(field.phi, size)
    scale_x = width / float(source_w)
    scale_y = height / float(source_h)
    return u * scale_x, v * scale_y, phi * mask


def _match_one(
    image0: torch.Tensor,
    image1: torch.Tensor,
    *,
    sigma_m: float,
    match_power: float,
    descriptor_smooth: float,
    sigmoid_slope: float,
    descriptor_regularizer: float,
) -> MatchField:
    height, width = image0.shape[-2:]
    zeros = torch.zeros_like(image0)
    centers_y = _centers(height, image0.device)
    centers_x = _centers(width, image0.device)
    if centers_y.numel() == 0 or centers_x.numel() == 0:
        return MatchField(zeros, zeros, zeros, zeros)

    desc0 = pixel_descriptors(image0, descriptor_smooth, sigmoid_slope, descriptor_regularizer)
    desc1 = pixel_descriptors(image1, descriptor_smooth, sigmoid_slope, descriptor_regularizer)
    nodes = _build_nodes(desc0, desc1, match_power, height, width)
    if nodes is None:
        score, target_y, target_x = _atomic_best(desc0, desc1, centers_y, centers_x)
    else:
        score, target_y, target_x = _backtrack(nodes, height, width)
    keep = _reciprocal(score, target_y, target_x, height, width)
    u, v, mask = _paint(target_y, target_x, keep, centers_y, centers_x, height, width)
    phi = _confidence(image0, image1, u, v, mask, sigma_m)
    return MatchField(u, v, mask, phi)


def _build_nodes(
    desc_ref: torch.Tensor,
    desc_tgt: torch.Tensor,
    power: float,
    height: int,
    width: int,
) -> list[_Node] | None:
    centers_y = _centers(height, desc_ref.device)
    centers_x = _centers(width, desc_ref.device)
    image_max = max(height, width)
    if centers_y.numel() < 2 or centers_x.numel() < 2 or image_max <= 4:
        return None
    patches = _reference_patches(desc_ref, centers_y, centers_x)
    pooled, argmax_y, argmax_x = _pooled_from_patches(
        desc_tgt, patches, int(centers_y.numel()), int(centers_x.numel())
    )
    gy = int(centers_y.numel())
    gx = int(centers_x.numel())
    nodes = [_Node(pooled, argmax_y, argmax_x, None, gy, gx)]
    patch = 4
    while patch < image_max and gy > 1 and gx > 1:
        score = aggregate_pooled(pooled, power)
        patch *= 2
        gy -= 1
        gx -= 1
        spatial_h, spatial_w = score.shape[-2:]
        stop = patch >= image_max or gy <= 1 or gx <= 1 or spatial_h < 2 or spatial_w < 2
        if stop:
            nodes.append(_Node(None, None, None, score, gy, gx))
            break
        pooled, argmax_y, argmax_x = _pool_subsample(score)
        pooled = pooled.clamp(min=0)
        nodes.append(_Node(pooled, argmax_y, argmax_x, score, gy, gx))
    if len(nodes) == 1:
        return None
    return nodes


def _backtrack(
    nodes: list[_Node],
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frontier: torch.Tensor | None = nodes[-1].score
    if frontier is None:
        raise RuntimeError("DeepMatching top level has no score")
    for index in range(len(nodes) - 2, 0, -1):
        frontier = _descend_volume(frontier, nodes[index])
    return _descend_atomic(frontier, nodes[0], height, width)


def _descend_volume(parent_score: torch.Tensor, child: _Node) -> torch.Tensor:
    if child.score is None or child.pooled is None or child.argmax_y is None or child.argmax_x is None:
        raise RuntimeError("DeepMatching mid level is missing its correlation")
    _check_child(parent_score, child)
    native_h, native_w = child.score.shape[-2:]
    child_gy, child_gx = child.pooled.shape[:2]
    pieces_index: list[torch.Tensor] = []
    pieces_value: list[torch.Tensor] = []
    pieces_y: list[torch.Tensor] = []
    pieces_x: list[torch.Tensor] = []
    for oy, ox in _QUADRANTS:
        iy, ix, native_y, native_x, valid, _qy, _qx = _quadrant_indices(
            parent_score, child, oy, ox
        )
        in_map = (native_y >= 0) & (native_y < native_h) & (native_x >= 0) & (native_x < native_w)
        valid = valid & in_map
        ny = native_y.clamp(0, native_h - 1)
        nx = native_x.clamp(0, native_w - 1)
        extra = child.score[iy, ix, ny, nx]
        total = torch.where(valid, parent_score + extra, torch.full_like(parent_score, _SENTINEL))
        pieces_index.append((iy * child_gx + ix).reshape(-1))
        pieces_value.append(total.reshape(-1))
        pieces_y.append(native_y.reshape(-1))
        pieces_x.append(native_x.reshape(-1))
    index = torch.cat(pieces_index)
    value = torch.cat(pieces_value)
    ys = torch.cat(pieces_y).clamp(0, native_h - 1)
    xs = torch.cat(pieces_x).clamp(0, native_w - 1)
    flat = index * (native_h * native_w) + ys * native_w + xs
    best = _scatter_max(flat, value, child_gy * child_gx * native_h * native_w)
    return best.view(child_gy, child_gx, native_h, native_w)


def _descend_atomic(
    parent_score: torch.Tensor,
    child: _Node,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if child.pooled is None or child.argmax_y is None or child.argmax_x is None:
        raise RuntimeError("DeepMatching atomic level is missing its argmax")
    _check_child(parent_score, child)
    child_gy, child_gx = child.pooled.shape[:2]
    pieces_index: list[torch.Tensor] = []
    pieces_value: list[torch.Tensor] = []
    pieces_y: list[torch.Tensor] = []
    pieces_x: list[torch.Tensor] = []
    for oy, ox in _QUADRANTS:
        iy, ix, native_y, native_x, valid, qy_c, qx_c = _quadrant_indices(
            parent_score, child, oy, ox
        )
        extra = child.pooled[iy, ix, qy_c, qx_c]
        in_image = (native_y >= 0) & (native_y < height) & (native_x >= 0) & (native_x < width)
        valid = valid & in_image
        total = torch.where(valid, parent_score + extra, torch.full_like(parent_score, _SENTINEL))
        pieces_index.append((iy * child_gx + ix).reshape(-1))
        pieces_value.append(total.reshape(-1))
        pieces_y.append(native_y.reshape(-1))
        pieces_x.append(native_x.reshape(-1))
    index = torch.cat(pieces_index)
    value = torch.cat(pieces_value)
    ys = torch.cat(pieces_y)
    xs = torch.cat(pieces_x)
    best, best_y, best_x = _scatter_argmax_coords(index, value, ys, xs, child_gy * child_gx)
    return best.view(child_gy, child_gx), best_y.view(child_gy, child_gx), best_x.view(child_gy, child_gx)


def _quadrant_indices(
    parent_score: torch.Tensor,
    child: _Node,
    oy: int,
    ox: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    parent_gy, parent_gx, pooled_h, pooled_w = parent_score.shape
    device = parent_score.device
    py = torch.arange(parent_gy, device=device).view(parent_gy, 1, 1, 1).expand(
        parent_gy, parent_gx, pooled_h, pooled_w
    )
    px = torch.arange(parent_gx, device=device).view(1, parent_gx, 1, 1).expand(
        parent_gy, parent_gx, pooled_h, pooled_w
    )
    sy = torch.arange(pooled_h, device=device).view(1, 1, pooled_h, 1).expand(
        parent_gy, parent_gx, pooled_h, pooled_w
    )
    sx = torch.arange(pooled_w, device=device).view(1, 1, 1, pooled_w).expand(
        parent_gy, parent_gx, pooled_h, pooled_w
    )
    iy = py + (0 if oy < 0 else 1)
    ix = px + (0 if ox < 0 else 1)
    # Same read as aggregate: output[p'] comes from the child at p' - o.
    qy = sy - oy
    qx = sx - ox
    valid = (qy >= 0) & (qy < pooled_h) & (qx >= 0) & (qx < pooled_w)
    qy_c = qy.clamp(0, pooled_h - 1)
    qx_c = qx.clamp(0, pooled_w - 1)
    assert child.argmax_y is not None and child.argmax_x is not None
    my = child.argmax_y[iy, ix, qy_c, qx_c]
    mx = child.argmax_x[iy, ix, qy_c, qx_c]
    native_y = 2 * qy + my
    native_x = 2 * qx + mx
    return iy, ix, native_y, native_x, valid, qy_c, qx_c


def _check_child(parent_score: torch.Tensor, child: _Node) -> None:
    assert child.pooled is not None
    if child.pooled.shape[0] != parent_score.shape[0] + 1 or child.pooled.shape[1] != parent_score.shape[1] + 1:
        raise RuntimeError(
            "DeepMatching parent/child grid mismatch: "
            f"parent {tuple(parent_score.shape[:2])} child {tuple(child.pooled.shape[:2])}"
        )
    if child.pooled.shape[-2:] != parent_score.shape[-2:]:
        raise RuntimeError(
            "DeepMatching spatial mismatch: "
            f"parent {tuple(parent_score.shape[-2:])} pooled {tuple(child.pooled.shape[-2:])}"
        )


def _atomic_best(
    desc_ref: torch.Tensor,
    desc_tgt: torch.Tensor,
    centers_y: torch.Tensor,
    centers_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    patches = _reference_patches(desc_ref, centers_y, centers_x)
    response = _correlate_chunks(desc_tgt, patches)
    gy = int(centers_y.numel())
    gx = int(centers_x.numel())
    _, height, width = response.shape
    flat = response.view(gy, gx, -1)
    score = flat.amax(dim=-1)
    span = flat.shape[-1]
    indices = torch.arange(span, device=flat.device).view(1, 1, span)
    close = flat >= score.unsqueeze(-1) - _TIE
    priority = torch.where(close, indices, torch.full_like(indices, span))
    index = priority.argmin(dim=-1)
    target_y = torch.div(index, width, rounding_mode="floor")
    target_x = index - target_y * width
    return score, target_y, target_x


def _reciprocal(
    score: torch.Tensor,
    target_y: torch.Tensor,
    target_x: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Keep a match only when it wins its target 4x4 block.

    Each atomic patch already holds its own best correspondence, which is the
    winner inside that non-overlapping 4x4 reference block.
    """
    inside = (
        (target_y >= 0)
        & (target_y < height)
        & (target_x >= 0)
        & (target_x < width)
        & (score > _SENTINEL * 0.5)
    )
    score = torch.where(inside, score, torch.full_like(score, _SENTINEL))
    blocks_y = max(1, (height + 3) // 4)
    blocks_x = max(1, (width + 3) // 4)
    block_y = torch.div(target_y.clamp(min=0), 4, rounding_mode="floor").clamp(0, blocks_y - 1)
    block_x = torch.div(target_x.clamp(min=0), 4, rounding_mode="floor").clamp(0, blocks_x - 1)
    flat_score = score.reshape(-1)
    flat_block = (block_y * blocks_x + block_x).reshape(-1)
    best = _scatter_max(flat_block, flat_score, blocks_y * blocks_x)
    patch_index = torch.arange(flat_score.numel(), device=score.device)
    winning = (flat_score > _SENTINEL * 0.5) & (flat_score >= best[flat_block] - _TIE)
    candidate = torch.where(winning, patch_index, torch.full_like(patch_index, patch_index.numel()))
    chosen = torch.full(
        (blocks_y * blocks_x,),
        patch_index.numel(),
        dtype=torch.long,
        device=score.device,
    )
    if patch_index.numel():
        chosen = chosen.scatter_reduce(0, flat_block, candidate, reduce="amin", include_self=True)
    is_best = patch_index == chosen[flat_block]
    return is_best.view_as(score)


def _paint(
    target_y: torch.Tensor,
    target_x: torch.Tensor,
    keep: torch.Tensor,
    centers_y: torch.Tensor,
    centers_x: torch.Tensor,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = target_y.device
    u = torch.zeros(1, 1, height, width, dtype=torch.float32, device=device)
    v = torch.zeros_like(u)
    mask = torch.zeros_like(u)
    rows, cols = torch.where(keep)
    if rows.numel() == 0:
        return u, v, mask
    cy = centers_y[rows]
    cx = centers_x[cols]
    du = (target_x[rows, cols] - cx).to(dtype=u.dtype)
    dv = (target_y[rows, cols] - cy).to(dtype=u.dtype)
    for oy in range(-2, 2):
        for ox in range(-2, 2):
            yy = (cy + oy).clamp(0, height - 1)
            xx = (cx + ox).clamp(0, width - 1)
            u[0, 0, yy, xx] = du
            v[0, 0, yy, xx] = dv
            mask[0, 0, yy, xx] = 1
    return u, v, mask


def _confidence(
    reference: torch.Tensor,
    target: torch.Tensor,
    flow_u: torch.Tensor,
    flow_v: torch.Tensor,
    mask: torch.Tensor,
    sigma_m: float,
) -> torch.Tensor:
    """ICCV confidence. ``lambda_tilde`` is 10 times the smaller eigenvalue."""
    gx = sobel_x(reference)
    gy = sobel_y(reference)
    a = _box_sum(gx * gx)
    b = _box_sum(gx * gy)
    c = _box_sum(gy * gy)
    smaller = 0.5 * (a + c - torch.sqrt((a - c) ** 2 + 4 * b * b)).clamp(min=0)
    lambda_tilde = 10.0 * smaller
    warped = _warp(target, flow_u, flow_v)
    warped_gx = _warp(sobel_x(target), flow_u, flow_v)
    warped_gy = _warp(sobel_y(target), flow_u, flow_v)
    delta = (reference - warped).abs() + (gx - warped_gx).abs() + (gy - warped_gy).abs()
    phi = lambda_tilde / (sigma_m * math.sqrt(2.0 * math.pi)) * torch.exp(-delta / (2.0 * sigma_m))
    return phi * mask


def _warp(image: torch.Tensor, flow_u: torch.Tensor, flow_v: torch.Tensor) -> torch.Tensor:
    height, width = image.shape[-2:]
    ys = torch.arange(height, device=image.device, dtype=image.dtype).view(1, 1, height, 1)
    xs = torch.arange(width, device=image.device, dtype=image.dtype).view(1, 1, 1, width)
    return remap_bilinear(image, xs + flow_u, ys + flow_v)


def _box_sum(image: torch.Tensor) -> torch.Tensor:
    padded = F.pad(image, (1, 1, 1, 1), mode="replicate")
    return F.avg_pool2d(padded, kernel_size=3, stride=1) * 9.0


def _centers(length: int, device: torch.device) -> torch.Tensor:
    if length < 4:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.arange(2, length - 1, 4, device=device)


def _match_size(height: int, width: int, max_side: int | None) -> tuple[int, int]:
    if max_side is None or max(height, width) <= max_side:
        return height, width
    scale = max_side / float(max(height, width))
    return max(4, int(round(height * scale))), max(4, int(round(width * scale)))


def _as_one(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 2:
        return image[None, None]
    if image.ndim == 3:
        return image[None]
    if image.ndim == 4 and image.shape[0] == 1 and image.shape[1] == 1:
        return image
    raise ValueError(f"expected one grayscale image, got {tuple(image.shape)}")


def _reference_patches(
    descriptor: torch.Tensor,
    centers_y: torch.Tensor,
    centers_x: torch.Tensor,
) -> torch.Tensor:
    gy = int(centers_y.numel())
    gx = int(centers_x.numel())
    oy = torch.arange(-2, 2, device=descriptor.device)
    ox = torch.arange(-2, 2, device=descriptor.device)
    yy = centers_y.view(gy, 1, 1, 1) + oy.view(1, 1, 4, 1)
    xx = centers_x.view(1, gx, 1, 1) + ox.view(1, 1, 1, 4)
    yy = yy.expand(gy, gx, 4, 4)
    xx = xx.expand(gy, gx, 4, 4)
    gathered = descriptor[0, :, yy, xx]
    return gathered.permute(1, 2, 0, 3, 4).reshape(gy * gx, 9, 4, 4).contiguous()


def _conv_patches(target_desc: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
    """Mean 4x4 dot product. Output ``(P, H, W)`` is centered on each pixel."""
    padded = F.pad(target_desc, (2, 1, 2, 1))
    response = F.conv2d(padded, patches)
    return response[0] / 16.0


def _correlate_chunks(target_desc: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
    parts = []
    for start in range(0, patches.shape[0], _CHUNK):
        parts.append(_conv_patches(target_desc, patches[start : start + _CHUNK]))
    return torch.cat(parts, dim=0)


def _pooled_from_patches(
    target_desc: torch.Tensor,
    patches: torch.Tensor,
    gy: int,
    gx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pooled_parts = []
    y_parts = []
    x_parts = []
    for start in range(0, patches.shape[0], _CHUNK):
        response = _conv_patches(target_desc, patches[start : start + _CHUNK])
        pooled, argmax_y, argmax_x = _pool_subsample(response)
        pooled_parts.append(pooled)
        y_parts.append(argmax_y)
        x_parts.append(argmax_x)
    pooled = torch.cat(pooled_parts, dim=0).clamp(min=0)
    argmax_y = torch.cat(y_parts, dim=0)
    argmax_x = torch.cat(x_parts, dim=0)
    pooled_h, pooled_w = pooled.shape[-2:]
    return (
        pooled.view(gy, gx, pooled_h, pooled_w),
        argmax_y.view(gy, gx, pooled_h, pooled_w),
        argmax_x.view(gy, gx, pooled_h, pooled_w),
    )


def _pool_subsample(
    response: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``P(C)(2 p') = max_{m in {-1,0,1}^2} C(2 p' + m)``, plus the argmax."""
    height, width = response.shape[-2:]
    pooled_h, pooled_w = height // 2, width // 2
    if pooled_h < 1 or pooled_w < 1:
        raise ValueError("cannot subsample a correlation map smaller than 2 pixels")
    flat = response.reshape(-1, height, width)
    values = []
    oys = []
    oxs = []
    for start in range(0, flat.shape[0], _CHUNK):
        value, oy, ox = _pool_block(flat[start : start + _CHUNK])
        values.append(value)
        oys.append(oy)
        oxs.append(ox)
    value = torch.cat(values, dim=0)
    oy = torch.cat(oys, dim=0)
    ox = torch.cat(oxs, dim=0)
    shape = (*response.shape[:-2], pooled_h, pooled_w)
    return value.reshape(shape), oy.reshape(shape), ox.reshape(shape)


def _pool_block(
    response: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = response.shape[-2:]
    pooled_h, pooled_w = height // 2, width // 2
    padded = F.pad(response, (1, 1, 1, 1), value=-1.0)
    samples = []
    for my in (-1, 0, 1):
        for mx in (-1, 0, 1):
            y0 = 1 + my
            x0 = 1 + mx
            samples.append(padded[..., y0 : y0 + 2 * pooled_h : 2, x0 : x0 + 2 * pooled_w : 2])
    stacked = torch.stack(samples, dim=0)
    value = stacked.amax(dim=0)
    # Lowest offset index among samples within ``_TIE`` of the max.
    close = stacked >= value.unsqueeze(0) - _TIE
    priority = torch.arange(stacked.shape[0], device=stacked.device).view(
        stacked.shape[0], *([1] * (stacked.ndim - 1))
    )
    priority = torch.where(close, priority, torch.full_like(priority, stacked.shape[0]))
    which = priority.argmin(dim=0)
    offset_y = which // 3 - 1
    offset_x = which % 3 - 1
    return value, offset_y.to(torch.long), offset_x.to(torch.long)


def _shift_with_mask(image: torch.Tensor, oy: int, ox: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``output[p'] = input[p' - o]``, with a 0 mask where that pixel is outside."""
    if oy == 0 and ox == 0:
        return image, torch.ones_like(image)
    shifted = torch.roll(image, shifts=(oy, ox), dims=(-2, -1))
    mask = torch.ones_like(image)
    if oy > 0:
        mask[..., :oy, :] = 0
    elif oy < 0:
        mask[..., oy:, :] = 0
    if ox > 0:
        mask[..., :, :ox] = 0
    elif ox < 0:
        mask[..., :, ox:] = 0
    return shifted * mask, mask


def _scatter_max(index: torch.Tensor, value: torch.Tensor, size: int) -> torch.Tensor:
    best = torch.full((size,), _SENTINEL, dtype=value.dtype, device=value.device)
    if index.numel() == 0:
        return best
    return best.scatter_reduce(0, index, value, reduce="amax", include_self=True)


def _scatter_argmax_coords(
    index: torch.Tensor,
    value: torch.Tensor,
    target_y: torch.Tensor,
    target_x: torch.Tensor,
    size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    best = _scatter_max(index, value, size)
    if index.numel() == 0:
        zeros = torch.zeros(size, dtype=torch.long, device=value.device)
        return best, zeros, zeros
    # ``scatter_`` with duplicate indices is not deterministic across devices.
    # Reduce with amin instead: highest score wins, and within ``_TIE`` the
    # smallest target coordinate wins.
    winning = value >= best[index] - _TIE
    offset = 100_000
    span = 2 * offset
    packed = (target_y.clamp(-offset + 1, offset - 1) + offset) * span + (
        target_x.clamp(-offset + 1, offset - 1) + offset
    )
    worst = span * span
    packed = torch.where(winning, packed, torch.full_like(packed, worst))
    chosen = torch.full((size,), worst, dtype=torch.long, device=value.device)
    chosen = chosen.scatter_reduce(0, index, packed, reduce="amin", include_self=True)
    best_y = torch.div(chosen, span, rounding_mode="floor") - offset
    best_x = chosen - (best_y + offset) * span - offset
    return best, best_y, best_x
